import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import os
import logging
from src.model_module import MidiGenModule
from src.tokenizer_module import get_tokenizer

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. 설정 및 디바이스 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"=== Titans(MAC) 작곡 시작 (Device: {device}) ===")
    
    # [중요] 추론 시에는 컴파일 기능을 끕니다 (오류 방지)
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False

    # 2. 토크나이저 로드
    tokenizer = get_tokenizer(cfg)
    log.info(f">> Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 3. 모델 수동 초기화 (load_from_checkpoint 대신 사용)
    # 이유: assign=True 옵션을 사용하고, 컴파일된 접두사를 제거하기 위함
    ckpt_dir = Path("checkpoints") / cfg.project_name
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        log.error(f"!! 체크포인트가 없습니다: {ckpt_dir}")
        return
    
    ckpt_path = str(ckpts[-1])
    log.info(f">> 최신 체크포인트 로드: {ckpt_path}")

    # (1) 모델 껍데기 생성 (컴파일 되지 않은 순정 상태)
    model_module = MidiGenModule(cfg, vocab_size=tokenizer.vocab_size)
    
    # (2) 체크포인트 파일 로드
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False) # weights_only=False for safety with older pytorch
    state_dict = checkpoint["state_dict"]

    # (3) 키 이름 정리 (컴파일된 모델의 '_orig_mod' 제거)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 'model._orig_mod.model.x' -> 'model.model.x' (MidiGenModule 구조에 맞춤)
        # 또는 'model.model.x'가 그대로 있을 수도 있음
        new_key = k.replace("model._orig_mod.", "model.") 
        new_state_dict[new_key] = v

    # (4) 가중치 로드 (핵심: assign=True)
    # assign=True는 텐서 값을 복사하는 게 아니라 포인터를 교체하므로
    # "shared memory location" 오류를 완벽하게 해결합니다.
    try:
        model_module.load_state_dict(new_state_dict, strict=False, assign=True)
        log.info(">> 모델 가중치 로드 성공 (assign=True 적용)")
    except Exception as e:
        log.error(f"!! 모델 로드 중 치명적 오류: {e}")
        return

    model_module.to(device)
    model_module.eval()
    
    # 4. 프롬프트 생성 (Conditioning)
    start_tokens = [tokenizer.start_token_id]
    
    # Global Header
    if 'Global_Header' in tokenizer.tokens_structure:
        start_tokens.append(tokenizer.tokens_structure['Global_Header'])
    
    # 작곡가 선택
    target_composer = cfg.target_composer
    if hasattr(tokenizer, 'composer_map') and target_composer in tokenizer.composer_map:
        composer_id = tokenizer.composer_vocab_start + tokenizer.composer_map[target_composer]
        start_tokens.append(composer_id)
        log.info(f">> 선택된 작곡가: {target_composer} (ID: {composer_id})")
    else:
        log.warning(f"!! 작곡가 '{target_composer}'를 찾을 수 없습니다.")

    # Narrative Stream
    if 'Narrative_Stream' in tokenizer.tokens_structure:
        start_tokens.append(tokenizer.tokens_structure['Narrative_Stream'])

    input_ids = torch.tensor([start_tokens], device=device).long()

    # 5. 생성 (Generation)
    log.info(f">> 생성 시작... 목표 길이: {cfg.target_length} 토큰")
    
    with torch.no_grad():
        generated_ids = model_module.model.generate(
            input_ids, 
            max_length=cfg.target_length,
            temperature=1.0,
            top_k=20,
            repetition_penalty=1.1,
        )

    # 6. 저장
    final_sequence = generated_ids[0].tolist()
    log.info(f">> 생성 완료! 총 길이: {len(final_sequence)}")

    final_midi_ids = [t for t in final_sequence if t < tokenizer.vocab_size]
    
    output_dir = Path("generated_output") / cfg.project_name
    os.makedirs(output_dir, exist_ok=True)
    
    save_filename = f"titans_{target_composer.replace(' ', '_')}_len{len(final_midi_ids)}.mid"
    save_path = output_dir / save_filename
    
    log.info(">> MIDI 변환 중...")
    try:
        tokenizer.decode(final_midi_ids, save_path)
        log.info(f"=== 🎹 작곡 완료! 저장됨: {save_path} ===")
    except Exception as e:
        log.error(f"!! 디코딩 실패: {e}")

if __name__ == "__main__":
    main()