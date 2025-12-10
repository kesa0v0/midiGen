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
    
    # tokenizer를 초기화하기 위해 data_module의 setup을 수동으로 호출하거나 prepare_data 이후에 접근해야 함
    # prepare_data는 main process에서 한 번만 호출되지만, setup은 각 프로세스에서 호출됨.
    # 여기서는 vocab_size를 얻기 위해 먼저 초기화가 필요함.
    data_module.prepare_data() 
    data_module.setup(stage="fit")

    # 실제 tokenizer의 vocab_size 가져오기
    actual_vocab_size = data_module.tokenizer.vocab_size
    print(f">> [Train] Tokenizer initialized. Vocab Size: {actual_vocab_size}")

    # 3. 모델 준비 (vocab_size 전달)
    model = MidiGenModule(cfg, vocab_size=actual_vocab_size)

    # 4. 콜백 설정
    checkpoint_dir = os.path.join("checkpoints", cfg.project_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config to checkpoint directory for easy reference
    from omegaconf import OmegaConf
    config_save_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(cfg, f)
    print(f">> [Train] Configuration saved to {config_save_path}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
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
        ckpts = sorted(Path(checkpoint_dir).glob("*.ckpt"), key=os.path.getmtime)
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
        log_every_n_steps=10,
        benchmark=True, # CUDNN 최적화 활성화 (속도 향상)
        accumulate_grad_batches=cfg.train.accumulate_grad_batches # 그래디언트 누적
    )

    # 6. 학습 시작
    if resume_ckpt:
        trainer.fit(model, data_module, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, data_module)

if __name__ == "__main__":
    main()