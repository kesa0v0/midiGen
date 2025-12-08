import hydra
from omegaconf import DictConfig
import pandas as pd
import os
from datasets import Dataset
from miditok import REMI, TokenizerConfig
from pathlib import Path

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"=== 데이터 전처리 시작: {cfg.project_name} ===")
    
    # 1. 파일 경로 확인
    if not os.path.exists(cfg.data.csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {cfg.data.csv_path}")

    # 2. 메타데이터 로드
    df = pd.read_csv(cfg.data.csv_path)
    
    # MIDI 파일 절대 경로 생성
    # MAESTRO 데이터셋은 CSV 안에 '2004/filename.midi' 형태로 들어있음
    df['full_path'] = df['midi_filename'].apply(lambda x: os.path.join(cfg.data.raw_path, x))
    
    # 파일 존재하는지 샘플 체크
    if not os.path.exists(df['full_path'].iloc[0]):
         raise FileNotFoundError(f"MIDI 파일을 찾을 수 없습니다. 경로를 확인하세요: {df['full_path'].iloc[0]}")

    print(f">> 총 데이터 개수: {len(df)}개")
    
    # 3. 토크나이저 설정
    tokenizer_config = TokenizerConfig(
        num_velocities=cfg.data.num_velocities, 
        use_chords=cfg.data.use_chords
    )
    tokenizer = REMI(tokenizer_config)

    # 4. 변환 함수
    def tokenize_midi(examples):
        token_ids_list = []
        for path in examples['full_path']:
            try:
                tokens = tokenizer(path)
                # 길이 제한 (설정된 길이보다 길면 자름, 짧으면 그대로 둠. 패딩은 학습 때 함)
                ids = tokens[0].ids[:cfg.data.max_seq_len]
                token_ids_list.append(ids)
            except Exception as e:
                print(f"Error: {e} | Path: {path}")
                token_ids_list.append([])
        return {'input_ids': token_ids_list}

    # 5. HuggingFace Dataset 변환 및 처리
    hf_dataset = Dataset.from_pandas(df[['split', 'full_path']])
    
    print(">> 토큰화 진행 중 (CPU 병렬 처리)...")
    # num_proc는 서버 CPU 코어 수에 맞춰 조절하세요 (예: 4, 8)
    tokenized_datasets = hf_dataset.map(
        tokenize_midi, 
        batched=True, 
        batch_size=32, 
        num_proc=4, 
        remove_columns=['full_path']
    )

    # 6. 저장
    print(f">> 데이터 저장 중: {cfg.data.processed_path}")
    tokenized_datasets.save_to_disk(cfg.data.processed_path)
    print("=== 전처리 완료! ===")

if __name__ == "__main__":
    main()