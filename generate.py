import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import os
import json
from src.model_module import MidiGenModule # Lightning Module 불러오기
from tqdm import tqdm
from src.tokenizer_module import get_tokenizer # Import the new tokenizer factory
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"=== Midigen Lightning 작곡 시작 (Device: {device}) ===")

    # 1. 토크나이저 로드 (새로운 추상화 사용)
    tokenizer = get_tokenizer(cfg)

    # 2. 작곡가 매핑 정보 로드
    composer_token_id = None
    target_composer = "Unknown"

    if os.path.exists("composer_map.json"):
        with open("composer_map.json", "r") as f:
            mapping_info = json.load(f)
        composer_to_id = mapping_info["composer_to_id"]
        
        # We need to ensure that the `base_vocab_size` from composer_map.json
        # is compatible with the current tokenizer's actual vocab_size.
        # If the new tokenizer has a different vocab size, the composer IDs might be incorrect.
        # For now, we'll assume the composer tokens are added after the base vocabulary.
        # This might need refinement if 'anticipation' handles composer info differently.
        base_vocab_size_from_map = mapping_info["base_vocab_size"]
        
        # Check if the current tokenizer's vocab size is at least the base_vocab_size from the map.
        if tokenizer.vocab_size < base_vocab_size_from_map:
            log.warning(
                "Tokenizer's current vocab size is smaller than base_vocab_size from composer_map.json. "
                "Composer tokens might be invalid. Proceeding with caution."
            )

        # config에서 target_composer 읽기
        target_composer = cfg.target_composer if hasattr(cfg, "target_composer") else "Frédéric Chopin"
        if target_composer not in composer_to_id:
            log.warning(f"Target composer '{target_composer}' not found in composer_map.json. Using first available composer.")
            target_composer = list(composer_to_id.keys())[0]
        
        # Assuming composer tokens are appended after the base vocabulary
        composer_token_id = tokenizer.vocab_size + composer_to_id[target_composer]
        log.info(f">> 선택된 작곡가: {target_composer} (ID: {composer_token_id})")
    else:
        log.warning("!! composer_map.json이 없습니다. 작곡가 정보 없이 진행합니다. 시작 토큰을 사용합니다.")
        # If no composer map, use the start token as the initial token for generation
        composer_token_id = tokenizer.start_token_id


    # 3. 모델 로드
    # checkpoints/{project_name} 폴더에서 가장 최신 .ckpt 파일을 찾음
    ckpt_dir = Path("checkpoints") / cfg.project_name
    if not ckpt_dir.exists():
        log.error(f"!! 체크포인트 디렉토리가 없습니다: {ckpt_dir}")
        return

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        log.error(f"!! '{ckpt_dir}'에 체크포인트(.ckpt)가 없습니다. 먼저 train.py를 실행하세요.")
        return
    
    ckpt_path = str(ckpts[-1])
    log.info(f">> 로드 중: {ckpt_dir} -> {ckpt_path}")

    # 모델 구조 + 가중치 자동 복구
    cfg.compile_model = False  # 생성 시에는 컴파일 비활성화
    # Load correct vocab size from tokenizer
    vocab_size = tokenizer.vocab_size
    model_module = MidiGenModule(cfg, vocab_size=vocab_size)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # 키 이름 변경 ('model._orig_mod.' -> 'model.')
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model._orig_mod.", "model.") # 컴파일된 접두사 제거
        new_state_dict[new_key] = v
        
    # 수정된 가중치 로드
    model_module.load_state_dict(new_state_dict)
    
    # 모델을 bfloat16으로 변환 (Flash Attention 2 필수 조건!)
    model_module.model.to(dtype=torch.bfloat16)
    model_module.to(device)
    model_module.eval()
    
    # 실제 GPT-2 모델 꺼내기
    model = model_module.model 


    # 슬라이딩 윈도우 생성 설정
    TARGET_LENGTH = cfg.target_length if hasattr(cfg, "target_length") and cfg.target_length and cfg.target_length <= cfg.tokenizer.max_seq_len * 16 else cfg.tokenizer.max_seq_len * 4  # 기본 4096
    CONTEXT_WINDOW = cfg.tokenizer.max_seq_len  # 모델이 한 번에 볼 수 있는 최대 길이 (학습 설정과 동일해야 함)
    NEW_TOKENS_PER_STEP = CONTEXT_WINDOW // 2  # 한 번에 생성할 길이 (CONTEXT_WINDOW의 절반 추천)

    log.info(f">> 목표 길이: {TARGET_LENGTH} 토큰 (슬라이딩 방식)")
    pbar = tqdm(total=TARGET_LENGTH, desc="작곡 중")
    
    # Initial sequence based on composer or start token
    if composer_token_id is not None:
        current_ids = [composer_token_id, tokenizer.bar_token_id]
    else:
        current_ids = [tokenizer.start_token_id, tokenizer.bar_token_id]

    full_generated_sequence = list(current_ids)
    pbar.update(len(full_generated_sequence))

    while len(full_generated_sequence) < TARGET_LENGTH:
        # 1) 입력 컨텍스트 준비 (가장 최근 토큰들만 잘라서 가져옴)
        max_context = CONTEXT_WINDOW - NEW_TOKENS_PER_STEP
        input_ids = full_generated_sequence[-max_context:]

        # 2) 생성
        with torch.no_grad():
            gen_len = len(input_ids) + NEW_TOKENS_PER_STEP
            output = model.generate(
                input_ids=torch.tensor([input_ids]).to(device),
                max_length=gen_len,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                no_repeat_ngram_size=32,
                pad_token_id=tokenizer.pad_token_id # Use tokenizer's pad_token_id
            )

        # 3) 새로운 토큰만 추출해서 전체 시퀀스에 추가
        new_tokens = output[0, len(input_ids):].tolist()
        full_generated_sequence.extend(new_tokens)
        pbar.update(len(new_tokens))

    pbar.close()

    # 5. 저장
    log.info("\n>> 변환 및 저장 중...")
    
    # Filter out tokens that are out of tokenizer's vocabulary range if necessary
    # This might be important if composer_token_id or other special tokens are handled outside the tokenizer's core vocab.
    final_midi_ids = [t for t in full_generated_sequence if t < tokenizer.vocab_size]

    output_dir = Path("generated_output") / cfg.project_name
    os.makedirs(output_dir, exist_ok=True)
    save_path = output_dir / f"output_{target_composer.replace(' ', '_')}_long.mid"
    
    # Use the abstracted tokenizer's decode method
    tokenizer.decode(final_midi_ids, Path(save_path))
    
    log.info(f"\n=== 🎹 긴 곡 작곡 완료! 저장됨: {save_path} ===")

if __name__ == "__main__":
    main()