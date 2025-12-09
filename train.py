import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.model_module import MidiGenModule
from src.data_module import MidiDataModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. 시드 고정 (재현성)
    pl.seed_everything(42)

    # 2. 모듈 준비
    data_module = MidiDataModule(cfg)
    model = MidiGenModule(cfg)

    # 3. 콜백 설정 (편의 기능)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="midigen-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    # 조기 종료 (Loss가 안 줄어들면 알아서 멈춤)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=3, 
        verbose=True
    )

    # 4. 트레이너 설정 (여기가 핵심)
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",     # GPU 있으면 알아서 씀
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed" if cfg.train.use_fp16 else 32, # 16비트 학습 지원
        log_every_n_steps=10
    )

    # 5. 학습 시작
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()