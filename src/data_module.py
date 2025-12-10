import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from datasets import load_from_disk, Dataset, DatasetDict
from miditok import REMI, TokenizerConfig
import pandas as pd
import os
from tqdm import tqdm

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 전처리된 데이터가 저장될 경로
        self.processed_path = cfg.data.processed_path

    # [핵심] preprocess.py의 내용을 여기로 옮겼습니다!
    # 이 함수는 학습 시작 전, 데이터를 준비할 때 '딱 한 번' 실행됩니다.
    def prepare_data(self):
        # 1. 이미 전처리된 데이터가 있으면 건너뜀 (캐싱)
        if os.path.exists(self.processed_path) and not self.cfg.train.force_preprocess:
            print(f">> [DataModule] 이미 전처리된 데이터가 있습니다: {self.processed_path}")
            return

        print(f">> [DataModule] 전처리된 데이터가 없습니다. 새로 만듭니다... (Augmentation 포함)")

        # --- 기존 preprocess.py 로직 시작 ---
        
        # 1) 메타데이터 로드
        df = pd.read_csv(self.cfg.data.csv_path)
        midi_paths = [os.path.join(self.cfg.data.raw_path, x) for x in df['midi_filename']]
        
        # 2) 토크나이저 설정 (표준 REMI)
        # config.yaml의 문자열 설정을 dict로 변환
        beat_res_dict = eval(self.cfg.data.beat_res) if isinstance(self.cfg.data.beat_res, str) else self.cfg.data.beat_res
        
        tokenizer_config = TokenizerConfig(
            num_velocities=self.cfg.data.num_velocities, 
            use_chords=True,
            use_tempos=True,
            beat_res=beat_res_dict
        )
        tokenizer = REMI(tokenizer_config)
        
        # 3) 데이터 증강 및 토큰화 (Sliding Window Chunking)
        all_token_ids = []
        seq_len = self.cfg.data.max_seq_len
        stride = seq_len  # 겹치지 않게 자름 (데이터 부족하면 seq_len // 2 로 변경)

        print(f">> 토큰화 및 청크 분할 시작 (총 {len(midi_paths)}곡)...")
        for path in tqdm(midi_paths):
            try:
                tokens = tokenizer(path)
                if not tokens: continue
                
                # 호환성 처리 (miditok 버전에 따라 다름)
                base_ids = tokens[0].ids if hasattr(tokens[0], 'ids') else tokens[0]

                # 긴 곡을 여러 개의 짧은 시퀀스로 자르기 (데이터 증강 효과)
                for i in range(0, len(base_ids) - seq_len, stride):
                    chunk = base_ids[i : i + seq_len]
                    # 길이가 너무 짧은 자투리는 버림 (노이즈 방지)
                    # [수정] 의미 없는 침묵 구간 필터링 (Silence Filtering)
                    # 청크 안에 'NoteOn'이나 'Pitch' 토큰이 최소 50개는 있어야 함
                    note_count = 0
                    for t in chunk:
                        # 토크나이저마다 이름이 다를 수 있으니 안전하게 확인
                        # 보통 REMI는 "Pitch_xx" 또는 "NoteOn_xx" 형태
                        token_str = tokenizer[t]
                        if token_str.startswith("Pitch") or token_str.startswith("NoteOn"):
                            note_count += 1
                    
                    # 1. 길이는 충분한가? (기존 조건)
                    # 2. 음표가 충분히 많은가? (추가된 조건 - 침묵 방지 핵심)
                    if len(chunk) >= seq_len // 2 and note_count > 50:
                        all_token_ids.append(chunk)
                        
            except Exception:
                continue

        print(f">> 생성된 시퀀스 개수: {len(all_token_ids)}")
        
        # 4) 데이터셋 저장
        full_dataset = Dataset.from_dict({'input_ids': all_token_ids})
        
        # Train/Val 분리 (9:1)
        split_dataset = full_dataset.train_test_split(test_size=0.1)
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        
        dataset_dict.save_to_disk(self.processed_path)
        print(f">> [DataModule] 전처리 완료 및 저장됨: {self.processed_path}")
        # --- 기존 preprocess.py 로직 끝 ---


    # 여기는 저장된 데이터를 불러오는 역할만 합니다.
    def setup(self, stage=None):
        dataset = load_from_disk(self.processed_path)
        self.train_ds = dataset['train']
        self.val_ds = dataset['validation']

    def collate_fn(self, batch):
        # 배치 내 길이를 맞추기 위한 패딩 (0으로 채움)
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        return padded

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.cfg.train.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.cfg.train.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=2
        )