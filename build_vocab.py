import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import os
import json
from tqdm import tqdm
from src.tokenizer_module import get_tokenizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info("=== Building Optimized Vocabulary for Anticipation Tokenizer ===")

    # 1. 데이터 경로 확인
    df = pd.read_csv(cfg.data.csv_path)
    midi_paths = [Path(cfg.data.raw_path) / x for x in df['midi_filename']]
    
    # jthickstun/anticipation 라이브러리 직접 호출을 위해 임포트
    try:
        from anticipation.convert import midi_to_events
    except ImportError:
        log.error("Anticipation library not found! Cannot build vocab.")
        return

    log.info(f"Scanning {len(midi_paths)} files to find ALL unique tokens...")

    unique_tokens = set()

    for path in tqdm(midi_paths):
        if not path.exists():
            continue
            
        try:
            # 순수 AMT 토큰 값 확인
            events = midi_to_events(str(path))
            if not events:
                continue
            
            unique_tokens.update(events)
            
        except Exception as e:
            # log.warning(f"Error processing {path}: {e}")
            pass

    sorted_tokens = sorted(list(unique_tokens))
    
    log.info(f"Scan Complete.")
    log.info(f"Total Unique Tokens Found: {len(sorted_tokens)}")
    log.info(f"Min Token ID: {sorted_tokens[0] if sorted_tokens else 'N/A'}")
    log.info(f"Max Token ID: {sorted_tokens[-1] if sorted_tokens else 'N/A'}")

    # 매핑 생성 (Original ID -> Compressed ID)
    # Compressed ID는 0부터 시작
    token_map = {original_id: i for i, original_id in enumerate(sorted_tokens)}
    
    # 저장
    vocab_file = Path("data/processed/anticipation_vocab_map.json")
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(vocab_file, "w") as f:
        json.dump(token_map, f, indent=4)
        
    log.info(f"Vocabulary map saved to {vocab_file}")
    log.info(f"Please update your Tokenizer to load this map!")

if __name__ == "__main__":
    main()
