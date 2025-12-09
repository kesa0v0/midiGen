import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_from_disk
import os
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from src.models import MidigenTitans

# 배치 내의 길이를 맞춰주는 함수 (Padding)
def collate_fn(batch):
    # batch는 [{'input_ids': [...]}, {'input_ids': [...]}] 형태
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    
    # 길이가 짧은 애들은 0으로 채워서 길이를 맞춤 (Padding)
    # batch_first=True -> (Batch, Seq_Len)
    padded_input = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return padded_input

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"=== Midigen 학습 시작: {cfg.model.type} ===")
    
    # 1. 장치 설정 (GPU 우선)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> 사용 장치: {device}")

    # 2. 데이터 로드 (아까 저장한 캐시 불러오기)
    if not os.path.exists(cfg.data.processed_path):
        raise FileNotFoundError("전처리된 데이터가 없습니다. python preprocess.py를 먼저 실행하세요.")
    
    dataset = load_from_disk(cfg.data.processed_path)
    # MAESTRO 데이터셋의 split 컬럼을 이용해 학습용만 필터링
    train_dataset = dataset.filter(lambda x: x['split'] == 'train')
    
    print(f">> 학습 데이터 개수: {len(train_dataset)}")

    # 3. DataLoader 준비
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,   # 윈도우라면 0이나 2로 줄여야 에러가 안 날 수도 있음 (Docker면 상관없음)
        pin_memory=True
    )

    # 4. 모델 초기화 (GPT-2 Baseline)
    if cfg.model.type == "gpt2":
        model_config = GPT2Config(
            vocab_size=cfg.data.vocab_size,
            n_positions=cfg.data.max_seq_len,
            n_embd=cfg.model.dim,
            n_layer=cfg.model.depth,
            n_head=cfg.model.heads,
            pad_token_id=0  # 패딩 토큰 ID
        )
        model = GPT2LMHeadModel(model_config).to(device)
    elif cfg.model.type == "titan":
        print(f">> Titans 모델을 로드합니다! (Chunk Size: {cfg.model.titan.chunk_size})")
        model = MidigenTitans(cfg).to(device)
    else:
        raise ValueError(f"지원하지 않는 모델: {cfg.model.type}")

    print(f">> 모델 생성 완료. 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 최적화 도구 (Optimizer)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)

    # 6. 학습 루프
    model.train()
    global_step = 0
    
    for epoch in range(cfg.train.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.train.epochs} 시작")
        
        for step, batch in enumerate(train_loader):
            inputs = batch.to(device)
            
            # GPT-2는 labels를 inputs와 똑같이 주면 내부적으로 Shift해서 Loss를 계산함
            # (다음 토큰 예측)
            outputs = model(input_ids=inputs, labels=inputs)
            loss = outputs.loss

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            
            # 로그 출력
            if step % 10 == 0:
                print(f"[Epoch {epoch+1}] Step {step} | Loss: {loss.item():.4f}")

        # 에폭마다 저장
        save_path = f"checkpoints/midigen_ep{epoch+1}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f">> 모델 저장됨: {save_path}")

if __name__ == "__main__":
    main()