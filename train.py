import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.model_module import MidiGenModule
from src.data_module import MidiDataModule
import torch
import os

# Tensor Core 활용 (속도 향상)
torch.set_float32_matmul_precision('medium')

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. 시드 고정
    pl.seed_everything(42)

    # 2. 데이터 모듈 준비 (여기서 전처리/증강이 자동으로 수행됨)
    data_module = MidiDataModule(cfg)

    # 3. 모델 준비
    model = MidiGenModule(cfg)

    # 4. 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="midigen-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,       # 가장 좋은 모델 2개만 저장
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=5,         # 5번 동안 성능 향상 없으면 조기 종료
        verbose=True
    )

    # 5. 트레이너 설정
    # 체크포인트 이어서 학습 경로 결정
    resume_ckpt = None
    ckpt_path_cfg = getattr(cfg.train, "resume_ckpt_path", "")
    if ckpt_path_cfg == "auto":
        from pathlib import Path
        ckpts = sorted(Path("checkpoints").glob("*.ckpt"), key=os.path.getmtime)
        if ckpts:
            resume_ckpt = str(ckpts[-1])
    elif ckpt_path_cfg:
        resume_ckpt = ckpt_path_cfg
    # 비워놓으면 None → 처음부터

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed",  # 혼합 정밀도 (메모리 절약 + 속도 UP)
        log_every_n_steps=10
    )

    # 6. 학습 시작
    if resume_ckpt:
        trainer.fit(model, data_module, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, data_module)

if __name__ == "__main__":
    main()