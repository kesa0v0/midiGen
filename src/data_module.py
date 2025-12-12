import json
import os
import glob
import random
import numpy as np
import pandas as pd
import itertools

from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
# from tqdm.contrib.concurrent import process_map # Removed to avoid memory overhead
from loguru import logger as log
import logging
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_from_disk, Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from src.tokenizer_module import get_tokenizer


log = logging.getLogger(__name__)

# Multiprocessing을 위한 헬퍼 함수 (클래스 외부에 정의해야 피클링 가능)
def _process_midi_file(item, cfg_dict, seq_len, stride, tokenizer_name):
    path, composer = item  # Unpacking
    
    cfg = OmegaConf.create(cfg_dict)
    
    # [Augmentation] Load shifts from config, default to [0] if not present
    augmentation_shifts = cfg.data.get("augmentation_shifts", [0])
    
    all_chunks = []
    
    try:
        tokenizer = get_tokenizer(cfg)
        
        # [Optimization] Load MIDI ONCE and get all augmented versions
        all_augmented_tokens = tokenizer.encode_with_augmentations(path, composer=composer, augment_shifts=augmentation_shifts)

        for tokens_ids in all_augmented_tokens:
            if not tokens_ids: continue

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
                    all_chunks.append(chunk)
        
        return all_chunks

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
        
        # [추가] 작곡가 매핑 생성 및 저장
        composers = sorted(df['canonical_composer'].unique())
        composer_to_id = {name: i for i, name in enumerate(composers)}
        
        # Tokenizer가 이 파일을 로드해서 Vocab을 확장함
        composer_map_path = os.path.join(self.cfg.paths.vocab, "composer_map.json")
        os.makedirs(os.path.dirname(composer_map_path), exist_ok=True)
        with open(composer_map_path, "w") as f:
            json.dump({
                "composer_to_id": composer_to_id,
                "count": len(composers)
            }, f, indent=4)
        log.info(f">> [DataModule] 작곡가 맵 저장됨 (총 {len(composers)}명) -> {composer_map_path}")

        # 2) 경로 및 작곡가 정보 준비
        midi_paths = [os.path.join(self.cfg.data.raw_path, x) for x in df['midi_filename']]
        midi_composers = df['canonical_composer'].tolist()
        
        # 병렬 처리를 위해 (경로, 작곡가) 튜플 리스트 생성
        process_items = list(zip(midi_paths, midi_composers))

        # 3) 토크나이저 설정
        # 메인 프로세스에서도 tokenizer 초기화 (vocab_size 확인 용도 등)
        self.tokenizer = get_tokenizer(self.cfg) # 이때 composer_map.json을 읽음

        # 4) 병렬 처리 설정
        seq_len = self.cfg.tokenizer.max_seq_len 
        stride = seq_len  
        tokenizer_name = self.cfg.tokenizer.name
        
        # OmegaConf를 dict로 변환 (Multiprocessing 피클링 문제 방지)
        from omegaconf import OmegaConf
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        log.info(f">> 병렬 토큰화 시작 (총 {len(midi_paths)}곡)...")
        
        # [Memory Optimization] Limit workers and chunksize
        safe_workers = min(os.cpu_count() or 1, 4) 
        
        # [True Streaming] Use Pool.imap instead of process_map
        # This avoids collecting all results in memory before creating the dataset.
        def streaming_generator():
            with Pool(processes=safe_workers) as pool:
                # partial function application
                process_func = partial(_process_midi_file, cfg_dict=cfg_dict, seq_len=seq_len, stride=stride, tokenizer_name=tokenizer_name)
                
                # imap yields results as they complete
                # chunksize=1 keeps memory footprint per worker low
                iterator = pool.imap(process_func, process_items, chunksize=1)
                
                # Wrap with tqdm for progress bar
                for result_chunks in tqdm(iterator, total=len(process_items), desc="Streaming Tokenization"):
                    for chunk in result_chunks:
                        yield {'input_ids': chunk}

        # 4) 데이터셋 저장
        # from_generator automatically handles streaming and caching to arrow file on disk
        log.info(">> Creating Dataset from streaming generator...")
        full_dataset = Dataset.from_generator(streaming_generator)

        log.info(f">> 생성된 시퀀스 개수: {len(full_dataset)}")

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