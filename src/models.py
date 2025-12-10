import torch
import torch.nn as nn
from titans_pytorch import MemoryAsContextTransformer
from transformers.modeling_outputs import CausalLMOutput

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
            seq_len_to_gen
            # temperature 등은 sample 함수 내부 기본값 사용 또는 필요시 **kwargs 파싱해서 전달
        )
        
        # 만약 입력보다 길이가 짧거나 같으면(오류 상황 등), 그냥 입력 반환
        if sampled.shape[1] <= current_len:
             return input_ids
             
        return sampled