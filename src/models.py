import torch
import torch.nn as nn
from titans_pytorch import MemoryAsContextTransformer
from transformers.modeling_outputs import CausalLMOutput
from mamba_ssm import Mamba2

class MidigenTitans(nn.Module):
    def __init__(self, cfg, vocab_size=None):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size if vocab_size is not None else cfg.data.vocab_size
        
        # 설정값 가져오기
        t_cfg = cfg.model.titan
        
        # Neural Memory 생성을 위한 설정값 딕셔너리 준비
        # (이전에 NeuralMemory 직접 만들 때 썼던 값들을 여기에 담습니다)
        neural_mem_kwargs = dict(
            dim_head = t_cfg.dim_head,
            heads = cfg.model.heads,
            activation = nn.SiLU() if t_cfg.activation == "silu" else nn.GELU(),
            momentum = t_cfg.momentum,
            default_model_kwargs = dict( # 메모리 내부 MLP 설정
                depth = t_cfg.memory_depth,
                expansion_factor = t_cfg.expansion_factor
            )
        )

        # [핵심] 완성형 Transformer 초기화
        self.model = MemoryAsContextTransformer(
            num_tokens = self.vocab_size,
            dim = cfg.model.dim,
            depth = cfg.model.depth,      # Transformer 레이어 수
            segment_len = t_cfg.chunk_size, # MAC 방식의 핵심: 세그먼트 길이
            
            # 메모리 토큰 설정
            num_persist_mem_tokens = t_cfg.num_persist_mem_tokens,
            num_longterm_mem_tokens = t_cfg.num_longterm_mem_tokens,
            
            # Neural Memory 세부 설정 전달
            neural_memory_kwargs = neural_mem_kwargs
        )

    def forward(self, input_ids, labels=None):
        # 학습 시: return_loss=True로 하면 내부에서 Loss까지 계산해서 줌
        if labels is not None:
            loss = self.model(input_ids, return_loss=True)
            # Pytorch Lightning 호환성을 위해 껍데기만 씌워서 리턴
            return CausalLMOutput(loss=loss, logits=None)
        
        # 추론 시: Logits 반환
        else:
            logits = self.model(input_ids)
            return CausalLMOutput(loss=None, logits=logits)

    def generate(self, input_ids, max_length=512, **kwargs):
        # MemoryAsContextTransformer에 내장된 최적화된 sample 함수 사용
        # (슬라이딩 윈도우, State 관리, Chunking 등이 내부적으로 자동 처리됨)
        
        current_len = input_ids.shape[1]
        seq_len_to_gen = max_length - current_len
        
        if seq_len_to_gen <= 0:
            return input_ids

        # sample()은 전체 시퀀스가 아니라 '생성된 부분'만 반환할 수도 있고, 
        # '전체'를 반환할 수도 있습니다 (라이브러리 버전에 따라 다름).
        # 보통 titans-pytorch는 전체 시퀀스를 반환하는 경향이 있습니다.
        sampled = self.model.sample(
            input_ids, 
            seq_len_to_gen,
            **kwargs
            # temperature 등은 sample 함수 내부 기본값 사용 또는 필요시 **kwargs 파싱해서 전달
        )
        
        # 만약 입력보다 길이가 짧거나 같으면(오류 상황 등), 그냥 입력 반환
        if sampled.shape[1] <= current_len:
             return input_ids
             
        return sampled


# Mamba Block 정의 (LayerNorm + Mamba + Residual)
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, headdim=8, layer_idx=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            headdim=headdim,
            layer_idx=layer_idx
        )
        
    def forward(self, x, inference_params=None):
        # Residual Connection
        return x + self.mamba(self.norm(x), inference_params=inference_params)

from mamba_ssm.utils.generation import InferenceParams

class MidigenMamba(nn.Module):
    def __init__(self, cfg, vocab_size=None):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.model.dim
        self.vocab_size = vocab_size if vocab_size is not None else cfg.data.vocab_size
        
        # 1. 임베딩
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        # Mamba는 Positional Embedding이 필수는 아니지만, 음악의 박자를 위해 넣는 것을 추천
        self.pos_emb = nn.Embedding(cfg.tokenizer.max_seq_len, self.dim)
        
        # 2. Mamba 레이어 스택
        mamba_cfg = cfg.model.mamba
        self.layers = nn.ModuleList([
            MambaBlock(
                dim=self.dim,
                d_state=mamba_cfg.d_state,
                d_conv=mamba_cfg.d_conv,
                expand=mamba_cfg.expand,
                headdim=mamba_cfg.head_dim,
                layer_idx=i
            ) for i in range(cfg.model.depth)
        ])
        
        # 3. 출력층
        self.norm_f = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, inference_params=None):
        b, n = input_ids.shape
        device = input_ids.device
        
        # 위치 임베딩 계산
        if inference_params is None:
            # 학습 모드 (전체 시퀀스)
            positions = torch.arange(n, device=device)
        else:
            # 추론 모드 (Step-by-step): 현재 시점의 위치 정보 사용
            # inference_params.seqlen_offset은 현재까지 처리된 길이
            positions = torch.arange(
                inference_params.seqlen_offset, 
                inference_params.seqlen_offset + n, 
                device=device
            )
            
        # [Safety] 위치 인덱스가 Positional Embedding 범위를 넘지 않도록 처리
        # 학습된 길이보다 길어질 경우 순환(modulo)시키는 방법 적용.
        max_pos_idx = self.pos_emb.num_embeddings
        positions = positions % max_pos_idx

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # Mamba 통과
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
            
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutput(loss=loss, logits=logits)

    # 생성 함수 (Autoregressive with State Caching)
    def generate(self, input_ids, max_length=512, temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.0, **kwargs):
        from tqdm import tqdm
        
        print(f"Debug: Starting Fast Mamba generation (Stateful). Current length: {input_ids.shape[1]}, Target: {max_length}")
        pbar = tqdm(total=max_length, initial=input_ids.shape[1], desc="Generating Tokens", dynamic_ncols=True)
        
        # 1. InferenceParams 초기화
        # max_seqlen: 최대 길이, max_batch_size: 배치 크기
        inference_params = InferenceParams(
            max_seqlen=max_length, 
            max_batch_size=input_ids.shape[0]
        )
        
        # 2. Prefill (초기 입력 처리)
        # 전체 프롬프트를 한 번에 넣어서 상태(State)를 초기화합니다.
        # 이때는 autoregressive하게 하나씩 할 필요 없이 한 번에 밀어넣습니다.
        with torch.no_grad():
            outputs = self(input_ids, inference_params=inference_params)
            next_token_logits = outputs.logits[:, -1, :]
            
            # 다음 토큰 샘플링을 위한 준비
            # (Prefill 단계에서는 마지막 토큰의 Logit만 필요함)
            
            # --- 첫 번째 토큰 샘플링 (Prefill 직후) ---
            # 후처리 (Sampling)
            if repetition_penalty != 1.0:
                # [Optimization] Vectorized Repetition Penalty (GPU)
                # 이전: CPU Loop & Set (Very Slow) -> 변경: Torch Scatter (Very Fast)
                score = torch.gather(next_token_logits, 1, input_ids)
                
                # if score < 0 then score * penalty else score / penalty
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                
                next_token_logits.scatter_(1, input_ids, score)

            # Top-K / Top-P / Temperature

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            if top_k > 0:
                v, _ = torch.topk(probs, top_k)
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, 1)
            
            # 결과 저장 및 입력 업데이트
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            pbar.update(1)
            
            # InferenceParams의 offset 업데이트 (Prefill 길이만큼)
            # mamba_ssm 버전에 따라 자동으로 업데이트 될 수도 있지만, 
            # 보통 forward 내부에서 처리되거나 명시적으로 관리해야 함.
            # self(..., inference_params) 호출 시 내부에서 seqlen_offset을 사용하므로,
            # 다음 step을 위해 올바르게 증가했는지 확인 필요.
            # Mamba 구현체는 보통 forward 호출 시 inference_params.seqlen_offset을 *읽고*, 
            # 내부 상태를 업데이트합니다. *하지만* offset 변수 자체를 자동으로 증가시키진 않을 수 있음.
            # 안전하게 수동 관리:
            inference_params.seqlen_offset += (input_ids.shape[1] - 1) - inference_params.seqlen_offset
            
            
        # 3. Decoding (Step-by-step)
        while input_ids.shape[1] < max_length:
            # 이전 단계에서 뽑은 'next_token' 하나만 입력으로 사용
            curr_input = next_token
            
            with torch.no_grad():
                # Step forward
                outputs = self(curr_input, inference_params=inference_params)
                next_token_logits = outputs.logits[:, -1, :] # (B, Vocab)
                
                # Sampling
                if repetition_penalty != 1.0:
                    # [Optimization] Vectorized Repetition Penalty (GPU)
                    score = torch.gather(next_token_logits, 1, input_ids)
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    next_token_logits.scatter_(1, input_ids, score)

                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                if top_k > 0:
                    v, _ = torch.topk(probs, top_k)
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, 1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Offset 증가
                inference_params.seqlen_offset += 1
                pbar.update(1)

        pbar.close()
        return input_ids