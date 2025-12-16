import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from src.models import MidigenTitans, MidigenMamba

class MidiGenModule(pl.LightningModule):
    def __init__(self, cfg, vocab_size=None):
        super().__init__()
        self.save_hyperparameters() # cfg 저장 (체크포인트에 자동 포함됨)
        self.cfg = cfg
        
        # Determine vocab_size: explicit argument > config > default error
        if vocab_size is not None:
            self.vocab_size = vocab_size
        elif hasattr(cfg, 'tokenizer') and hasattr(cfg.tokenizer, 'vocab_size'):
             # Fallback for backward compatibility or if passed in config (though discouraged)
            self.vocab_size = cfg.tokenizer.vocab_size
        elif hasattr(cfg, 'data') and hasattr(cfg.data, 'vocab_size'):
             # Legacy fallback
            self.vocab_size = cfg.data.vocab_size
        else:
             # Default to a safe placeholder if mostly testing, but warn
             # In production, this should likely raise an error.
             print("Warning: vocab_size not provided. Using default 500.")
             self.vocab_size = 500

        # 1. 모델 정의
        if cfg.model.type == "gpt2":
            config = GPT2Config(
                vocab_size=self.vocab_size,
                n_positions=cfg.tokenizer.max_seq_len, # Updated to cfg.tokenizer
                n_embd=cfg.model.dim,
                n_layer=cfg.model.depth,
                n_head=cfg.model.heads,
                pad_token_id=0,
                attn_implementation="flash_attention_2"
            )
            self.model = GPT2LMHeadModel(config)
        elif cfg.model.type == "titan":
            # Titans 모델 초기화
            # vocab_size를 명시적으로 전달하여 정합성 보장
            self.model = MidigenTitans(cfg, vocab_size=self.vocab_size)
        elif cfg.model.type == "mamba":
            self.model = MidigenMamba(cfg, vocab_size=self.vocab_size)
        else:
            raise NotImplementedError(f"Model type '{cfg.model.type}' not implemented")        


        compile_model = getattr(cfg, "compile_model", False)
        if compile_model:
            try:
                # mode="reduce-overhead": CUDA 그래프 등을 활용해 CPU 오버헤드 최소화 (학습 속도 향상)
                # "default" 모드로 변경하여 호환성 확보
                self.model = torch.compile(self.model, mode="default")
            except Exception as e:
                print(f"!! torch.compile 실패 (무시하고 진행합니다): {e}")

    def forward(self, input_ids):
        return self.model(input_ids, labels=input_ids)

    # 2. 학습 스텝 (train.py의 루프 안쪽 내용)
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        # 로그 자동 기록 (WandB 등에 바로 뜸)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # 3. 검증 스텝 (Validation)
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    # 4. 옵티마이저 설정
    def configure_optimizers(self):
        # 1. 옵티마이저 설정 (AdamW 추천)
        # fused=True: PyTorch 2.0+에서 GPU 연산 속도 향상
        optimizer = AdamW(self.parameters(), lr=self.cfg.train.lr, fused=False)
        
        # 2. 스케줄러 설정 (Cosine Annealing)
        # T_max: 보통 전체 에폭(epochs) 수로 설정합니다.
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.train.epochs, 
            eta_min=1e-6  # 학습률이 0이 되지 않도록 최소값 설정
        )
        
        # 3. Lightning에 전달할 설정 딕셔너리 반환
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 에폭 단위로 스케줄러 실행 (step으로 하려면 "step")
                "monitor": "val_loss" # ReduceLROnPlateau 같은 거 쓸 때 필요
            }
        }