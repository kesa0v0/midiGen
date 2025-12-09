import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel
from miditok import REMI

class MidiGenModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters() # cfg 저장 (체크포인트에 자동 포함됨)
        self.cfg = cfg
        
        # 1. 모델 정의
        if cfg.model.type == "gpt2":
            config = GPT2Config(
                vocab_size=cfg.data.vocab_size,
                n_positions=cfg.data.max_seq_len,
                n_embd=cfg.model.dim,
                n_layer=cfg.model.depth,
                n_head=cfg.model.heads,
                pad_token_id=0
            )
            self.model = GPT2LMHeadModel(config)
        else:
            # Titans 등 다른 모델 확장 가능
            raise NotImplementedError("GPT2 외 모델은 아직 미구현")

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
        return AdamW(self.parameters(), lr=self.cfg.train.lr)