import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from datasets import load_from_disk, Dataset, DatasetDict
import pandas as pd
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from src.tokenizer_module import get_tokenizer # Import the new tokenizer factory
import logging
from functools import partial

log = logging.getLogger(__name__)

# Multiprocessing을 위한 헬퍼 함수 (클래스 외부에 정의해야 피클링 가능)
def _process_midi_file(path, cfg_dict, seq_len, stride, tokenizer_name):
    # 각 프로세스마다 tokenizer 새로 생성 (필수)
    # OmegaConf 객체는 피클링이 안 될 수 있으므로 dict로 받아서 다시 변환하거나
    # 여기서 필요한 설정만 사용하는 것이 좋음. 편의상 cfg 전체를 넘기되,
    # 실제로는 get_tokenizer가 OmegaConf를 기대하므로 여기서 변환 처리 필요.
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_dict)
    
    try:
        tokenizer = get_tokenizer(cfg)
        tokens_ids = tokenizer.encode(path)
        if not tokens_ids: return []

        chunks = []
        # 긴 곡을 여러 개의 짧은 시퀀스로 자르기 (데이터 증강 효과)
        for i in range(0, len(tokens_ids) - seq_len, stride):
            chunk = tokens_ids[i : i + seq_len]
            
            # [수정] 의미 없는 침묵 구간 필터링 (Silence Filtering)
            note_count = 0
            # REMI tokenizer specific check logic
            if tokenizer_name == "remi":
                 for t in chunk:
                    token_str = tokenizer.tokenizer[t] 
                    if token_str.startswith("Pitch") or token_str.startswith("NoteOn"):
                        note_count += 1
            else: 
                # Placeholder for anticipation
                note_count += 100 

            # 1. 길이는 충분한가?
            # 2. 음표가 충분히 많은가?
            if len(chunk) >= seq_len // 2 and note_count > 50:
                chunks.append(chunk)
        
        return chunks

    except Exception as e:
        # log.warning(f"Failed to process {path}: {e}") # 너무 많이 뜨면 시끄러움
        return []

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 전처리된 데이터가 저장될 경로
        self.processed_path = cfg.data.processed_path
        self.tokenizer = None # Initialize tokenizer here

    # [핵심] preprocess.py의 내용을 여기로 옮겼습니다!
    # 이 함수는 학습 시작 전, 데이터를 준비할 때 '딱 한 번' 실행됩니다.
    def prepare_data(self):
        # 1. 이미 전처리된 데이터가 있으면 건너뜀 (캐싱)
        if os.path.exists(self.processed_path) and not self.cfg.train.force_preprocess:
            log.info(f">> [DataModule] 이미 전처리된 데이터가 있습니다: {self.processed_path}")
            return

        log.info(f">> [DataModule] 전처리된 데이터가 없습니다. 새로 만듭니다... (Augmentation 포함)")

        # --- 기존 preprocess.py 로직 시작 ---

        # 1) 메타데이터 로드
        df = pd.read_csv(self.cfg.data.csv_path)
        midi_paths = [os.path.join(self.cfg.data.raw_path, x) for x in df['midi_filename']]

        # 2) 토크나이저 설정
        # 메인 프로세스에서도 tokenizer 초기화 (vocab_size 확인 용도 등)
        self.tokenizer = get_tokenizer(self.cfg)

        # 3) 병렬 처리 설정
        seq_len = self.cfg.tokenizer.max_seq_len 
        stride = seq_len  
        tokenizer_name = self.cfg.tokenizer.name
        
        # OmegaConf를 dict로 변환 (Multiprocessing 피클링 문제 방지)
        from omegaconf import OmegaConf
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        log.info(f">> 병렬 토큰화 시작 (총 {len(midi_paths)}곡, CPU 코어 활용)...")
        
        # process_map을 사용하여 병렬 처리 실행
        # max_workers는 CPU 코어 수만큼 자동 할당됨
        results = process_map(
            partial(_process_midi_file, cfg_dict=cfg_dict, seq_len=seq_len, stride=stride, tokenizer_name=tokenizer_name),
            midi_paths,
            chunksize=10,
            desc="Tokenizing MIDI files"
        )

        # 결과 리스트 평탄화 (Flatten)
        all_token_ids = [chunk for result in results for chunk in result]

        log.info(f">> 생성된 시퀀스 개수: {len(all_token_ids)}")

        # 4) 데이터셋 저장
        full_dataset = Dataset.from_dict({'input_ids': all_token_ids})

        # Train/Val 분리 (9:1)
        split_dataset = full_dataset.train_test_split(test_size=0.1)
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })

        dataset_dict.save_to_disk(self.processed_path)
        log.info(f">> [DataModule] 전처리 완료 및 저장됨: {self.processed_path}")
        # --- 기존 preprocess.py 로직 끝 ---


    # 여기는 저장된 데이터를 불러오는 역할만 합니다.
    def setup(self, stage=None):
        dataset = load_from_disk(self.processed_path)
        self.train_ds = dataset['train']
        self.val_ds = dataset['validation']
        # Initialize tokenizer if not already done (e.g., if prepare_data was skipped due to caching)
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer(self.cfg)

    def collate_fn(self, batch):
        # 배치 내 길이를 맞추기 위한 패딩 (0으로 채움)
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        # Use the tokenizer's pad_token_id for padding
        padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return padded

    def train_dataloader(self):
        # CPU 코어 수 기반으로 워커 수 자동 설정 (최대 8개)
        num_workers = min(os.cpu_count() // 2, 8) if os.cpu_count() else 4
        
        return DataLoader(
            self.train_ds, 
            batch_size=self.cfg.train.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=True, # 워커 프로세스 유지 (오버헤드 감소)
            pin_memory=True
        )

    def val_dataloader(self):
        num_workers = min(os.cpu_count() // 2, 4) if os.cpu_count() else 2
        return DataLoader(
            self.val_ds, 
            batch_size=self.cfg.train.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=True # 검증도 유지하면 좋음
        )