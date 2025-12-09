import torch
import torch.nn as nn
from titans_pytorch import NeuralMemory


class MidigenTitans(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.model.dim
        self.vocab_size = cfg.data.vocab_size
        
        # 1. 임베딩
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(cfg.data.max_seq_len, self.dim)
        
        # 2. Titans Neural Memory
        self.titans = NeuralMemory(
            dim = self.dim,
            chunk_size = cfg.model.titan.chunk_size,
            dim_head = 64,
            heads = cfg.model.heads,
            activation = nn.SiLU() 
        )
        
        # 3. 정규화 및 출력
        self.norm = nn.LayerNorm(self.dim)
        self.to_logits = nn.Linear(self.dim, self.vocab_size)

    def forward(self, input_ids, labels=None):
        b, n = input_ids.shape
        device = input_ids.device
        
        # 임베딩
        positions = torch.arange(n, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # Titans 통과 (튜플 반환 대응)
        titan_out = self.titans(x)
        if isinstance(titan_out, tuple):
            x = titan_out[0]
        else:
            x = titan_out
        
        x = self.norm(x)
        logits = self.to_logits(x) # (Batch, Seq, Vocab)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            from transformers.modeling_outputs import CausalLMOutput
            return CausalLMOutput(loss=loss, logits=logits)
            
        return logits

    # [핵심 수정] generate 함수가 _simple_generate를 확실하게 호출하도록 연결
    def generate(self, input_ids, max_length=512, **kwargs):
        return self._simple_generate(input_ids, max_length=max_length, **kwargs)
    

    def _simple_generate(self, input_ids, max_length=1024, temperature=0.9, top_k=40, repetition_penalty=1.0, **kwargs):
        # [수정] 모델의 물리적 한계 길이
        model_max_len = self.pos_emb.num_embeddings
        final_len = min(max_length, model_max_len)
        
        while input_ids.shape[1] < final_len:
            # 1. 모델 예측
            logits = self(input_ids) 
            
            if isinstance(logits, dict) or hasattr(logits, 'logits'):
                next_token_logits = logits.logits[:, -1, :]
            else:
                next_token_logits = logits[:, -1, :]
            
            # 2. [수정] EOS(0번) 토큰 원천 봉쇄
            # AI가 "이제 그만할래(0)"라고 말 못하게 입을 막습니다.
            next_token_logits[:, 0] = -float('inf')

            # 3. Repetition Penalty
            # (음악에서는 1.0으로 끄는 것을 추천하지만, 파라미터는 살려둠)
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    previous_tokens = set(input_ids[i].tolist())
                    for token in previous_tokens:
                        if next_token_logits[i, token] < 0:
                            next_token_logits[i, token] *= repetition_penalty
                        else:
                            next_token_logits[i, token] /= repetition_penalty

            # 4. Sampling
            # Temperature를 높여서(0.9~1.0) 좀 더 과감하게 시도하게 함
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            
            # Top-k
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            next_token_index = torch.multinomial(top_k_probs, 1)
            next_token = torch.gather(top_k_indices, -1, next_token_index)
            
            # 5. 이어 붙이기
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # 종료 조건 없음 (무조건 꽉 채울 때까지 달림)
                
        return input_ids