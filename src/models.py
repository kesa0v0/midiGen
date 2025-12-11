import torch
import torch.nn as nn
from titans_pytorch import MemoryAsContextTransformer
from transformers.modeling_outputs import CausalLMOutput
from mamba_ssm import Mamba

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
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        
    def forward(self, x):
        # Residual Connection
        return x + self.mamba(self.norm(x))

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
                expand=mamba_cfg.expand
            ) for _ in range(cfg.model.depth)
        ])
        
        # 3. 출력층
        self.norm_f = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        b, n = input_ids.shape
        device = input_ids.device
        
        # 위치 임베딩
        positions = torch.arange(n, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # Mamba 통과
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutput(loss=loss, logits=logits)

    # 생성 함수 (Autoregressive)
    def generate(self, input_ids, max_length=512, temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.0, **kwargs):
        # Mamba는 전체 시퀀스를 다시 넣어도 연산량이 선형(Linear)이라 빠릅니다.
        # 최적화된 step-by-step inference도 가능하지만, 구현 편의를 위해 일단 전체 입력 방식을 씁니다.
        
        model_max_len = self.pos_emb.num_embeddings
        
        while input_ids.shape[1] < max_length:
            # 입력 길이 제한 (Positional Embedding 한계 때문)
            curr_input = input_ids if input_ids.shape[1] < model_max_len else input_ids[:, -model_max_len:]
            
            outputs = self(curr_input)
            next_token_logits = outputs.logits[:, -1, :]
            
            # 후처리 (Sampling)
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    for token in set(input_ids[i].tolist()):
                        if next_token_logits[i, token] < 0:
                            next_token_logits[i, token] *= repetition_penalty
                        else:
                            next_token_logits[i, token] /= repetition_penalty

            # Top-K / Top-P / Temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # (생략 가능) 간단한 Top-K 구현
            if top_k > 0:
                v, _ = torch.topk(probs, top_k)
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
        return input_ids