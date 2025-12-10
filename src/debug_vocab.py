import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
from src.tokenizer_module import get_tokenizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info("=== Vocab Size Debugging Start ===")

    # 1. 데이터 경로 확인
    df = pd.read_csv(cfg.data.csv_path)
    midi_paths = [Path(cfg.data.raw_path) / x for x in df['midi_filename']]
    
    # 2. 토크나이저 초기화
    tokenizer = get_tokenizer(cfg)
    log.info(f"Tokenizer: {cfg.tokenizer.name}")
    log.info(f"Configured Vocab Size: {tokenizer.vocab_size}")
    
    if hasattr(tokenizer, 'amt_vocab_start'):
        log.info(f"AMT Vocab Offset: {tokenizer.amt_vocab_start}")
        log.info(f"AMT Base Vocab Size: {tokenizer._base_amt_vocab_size}")

    # 3. 샘플링 검사 (속도를 위해 일부만 검사하거나 전체 검사)
    max_token_found = -1
    min_token_found = float('inf')
    out_of_bound_count = 0
    total_tokens = 0
    
    # jthickstun/anticipation 라이브러리 직접 호출을 위해 임포트
    try:
        from anticipation.convert import midi_to_events
    except ImportError:
        log.error("Anticipation library not found!")
        return

    log.info(f"Scanning {len(midi_paths)} files...")

    for i, path in enumerate(tqdm(midi_paths)):
        if not path.exists():
            continue
            
        try:
            # 순수 AMT 토큰 값 확인 (오프셋 적용 전)
            events = midi_to_events(str(path))
            if not events:
                continue
            
            local_max = max(events)
            local_min = min(events)
            
            # 오프셋 적용 후의 예상 값
            shifted_max = local_max + tokenizer.amt_vocab_start
            
            if shifted_max > max_token_found:
                max_token_found = shifted_max
                log.info(f"[New Max] File: {path.name}, Raw Max: {local_max}, Shifted Max: {shifted_max}")
            
            if shifted_max >= tokenizer.vocab_size:
                out_of_bound_count += 1
                log.warning(f"!! Out of Bound Detected in {path.name}: Shifted Max {shifted_max} >= Vocab Size {tokenizer.vocab_size}")
            
            if local_min < min_token_found:
                min_token_found = local_min
                
            total_tokens += len(events)
            
        except Exception as e:
            # log.warning(f"Error processing {path}: {e}")
            pass

    log.info("=== Summary ===")
    log.info(f"Files Processed: {len(midi_paths)}")
    log.info(f"Global Max Token (Shifted): {max_token_found}")
    log.info(f"Current Configured Vocab Size: {tokenizer.vocab_size}")
    
    if max_token_found >= tokenizer.vocab_size:
        log.error(f"CRITICAL: Found tokens exceeding vocab size! Need to increase vocab size to at least {max_token_found + 1}")
    else:
        log.info("SAFE: All tokens are within vocabulary range.")

if __name__ == "__main__":
    main()