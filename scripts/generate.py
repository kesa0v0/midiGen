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
    model_type = cfg.model.type.upper() if hasattr(cfg, "model") else "UNKNOWN"
    log.info(f"=== {model_type} 작곡 시작 (Device: {device}) ===")
    
    # [중요] 추론 시에는 컴파일 기능을 끕니다 (오류 방지)
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False

    # 2. 토크나이저 로드
    tokenizer = get_tokenizer(cfg)
    log.info(f">> Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 3. 모델 수동 초기화 (load_from_checkpoint 대신 사용)
    # 이유: assign=True 옵션을 사용하고, 컴파일된 접두사를 제거하기 위함
    ckpt_dir = Path(cfg.paths.checkpoints)
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
    
    PRIMING_MIDI_PATH = "data/raw/maestro-v3.0.0/2004/MIDI-Unprocessed_Schubert4-6_MID--AUDIO_10_R2_2018_wav.midi"
    
    start_tokens = [tokenizer.start_token_id]
    
    # (1) Global Header (Anticipation 호환성 유지)
    if hasattr(tokenizer, 'tokens_structure') and 'Global_Header' in tokenizer.tokens_structure:
        start_tokens.append(tokenizer.tokens_structure['Global_Header'])
    
    # (2) 작곡가 선택 (REMI에서는 무시됨)
    target_composer = cfg.target_composer
    if hasattr(tokenizer, 'composer_map') and target_composer in tokenizer.composer_map:
        composer_id = tokenizer.composer_vocab_start + tokenizer.composer_map[target_composer]
        start_tokens.append(composer_id)
        log.info(f">> 선택된 작곡가: {target_composer} (ID: {composer_id})")
    
    # (3) Narrative Stream (Anticipation 호환성 유지)
    if hasattr(tokenizer, 'tokens_structure') and 'Narrative_Stream' in tokenizer.tokens_structure:
        start_tokens.append(tokenizer.tokens_structure['Narrative_Stream'])

    # (4) [핵심] 프라이밍(Priming): 반주 지옥 탈출을 위한 강제 주입
    priming_tokens = []
    if os.path.exists(PRIMING_MIDI_PATH):
        log.info(f">> 프라이밍(Priming) 시도: {PRIMING_MIDI_PATH} 참고 중...")
        try:
            # MIDI 파일을 토큰으로 변환 (REMI 토크나이저 사용)
            full_tokens = tokenizer.encode(PRIMING_MIDI_PATH)
            
            # 앞에서부터 200개 정도 자르기 (도입부~초반 멜로디)
            # 너무 길면 생성할 공간이 줄어드니 적당히 자릅니다.
            priming_tokens = full_tokens[:200]
            
            # 중복된 시작 토큰 제거
            if priming_tokens and priming_tokens[0] == tokenizer.start_token_id:
                priming_tokens = priming_tokens[1:]
                
            log.info(f">> 프라이밍 토큰 {len(priming_tokens)}개 주입 완료! (반주 패턴 탈출 유도)")
        except Exception as e:
            log.warning(f"!! 프라이밍 파일 로드 실패 (기본 모드로 시작합니다): {e}")
    else:
        # 경로가 틀렸거나 파일이 없으면 경고 메시지를 띄우고 그냥 진행합니다.
        log.warning(f"!! 프라이밍 MIDI 파일을 찾을 수 없습니다: {PRIMING_MIDI_PATH}")
        log.warning("   -> 경로를 확인하거나, 그냥 깡통(BOS) 상태로 시작합니다.")

    # 최종 입력: [Start] + [Metadata] + [Priming(멜로디)]
    final_input_tokens = start_tokens + priming_tokens
    input_ids = torch.tensor([final_input_tokens], device=device).long()

    # 5. 생성 (Generation)
    # 프라이밍 길이만큼 목표 길이를 늘려줍니다.
    total_target_len = cfg.target_length + len(priming_tokens)
    log.info(f">> 생성 시작... (입력: {len(final_input_tokens)} -> 목표: {total_target_len})")
    
    with torch.no_grad():
        generated_ids = model_module.model.generate(
            input_ids, 
            max_length=total_target_len, 
            temperature=1.0,     # 창의성 1.0 유지
            top_k=80,            # 80 유지 (다양성 확보)
            top_p=0.95,          # 0.95 유지
            # repetition_penalty는 지원하지 않으므로 생략
        )

    # 6. 저장
    final_sequence = generated_ids[0].tolist()
    log.info(f">> 생성 완료! 총 길이: {len(final_sequence)}")

    final_midi_ids = [t for t in final_sequence if t < tokenizer.vocab_size]
    
    output_dir = Path(cfg.paths.outputs)
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