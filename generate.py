import hydra
from omegaconf import DictConfig
import torch
from miditok import REMI, TokenizerConfig
from pathlib import Path
import os
import json
from src.model_module import MidiGenModule # Lightning Module 불러오기
from tqdm import tqdm

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Midigen Lightning 작곡 시작 (Device: {device}) ===")

    # 1. 작곡가 매핑 정보 로드
    if not os.path.exists("composer_map.json"):
        print("!! composer_map.json이 없습니다. 랜덤으로 진행합니다.")
        composer_token_id = 0
        target_composer = "Unknown"
    else:
        with open("composer_map.json", "r") as f:
            mapping_info = json.load(f)
        composer_to_id = mapping_info["composer_to_id"]
        base_vocab_size = mapping_info["base_vocab_size"]
        
        target_composer = "Frédéric Chopin" 
        if target_composer not in composer_to_id:
            target_composer = list(composer_to_id.keys())[0]
        
        composer_token_id = base_vocab_size + composer_to_id[target_composer]
        print(f">> 선택된 작곡가: {target_composer} (ID: {composer_token_id})")

    # 2. 토크나이저 로드 (표준 REMI)
    beat_res_dict = eval(cfg.data.beat_res) if isinstance(cfg.data.beat_res, str) else cfg.data.beat_res
    tokenizer_config = TokenizerConfig(
        num_velocities=cfg.data.num_velocities, 
        use_chords=True,
        use_tempos=True,
        beat_res=beat_res_dict
    )
    tokenizer = REMI(tokenizer_config)

    # 3. 모델 로드
    # checkpoints 폴더에서 가장 최신 .ckpt 파일을 찾음
    ckpts = sorted(Path("checkpoints").glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        print("!! 체크포인트(.ckpt)가 없습니다. 먼저 train.py를 실행하세요.")
        return
    
    ckpt_path = str(ckpts[-1])
    print(f">> 로드 중: {ckpt_path}")

    # 모델 구조 + 가중치 자동 복구
    # model_module = MidiGenModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    cfg.compile_model = False  # 생성 시에는 컴파일 비활성화
    model_module = MidiGenModule(cfg)

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
    TARGET_LENGTH = cfg.target_length if cfg.target_length or cfg.target_length <= 1024 else 1024  # 목표 곡 길이 (토큰 수). 약 3~4분 분량
    CONTEXT_WINDOW = cfg.data.max_seq_len  # 모델이 한 번에 볼 수 있는 최대 길이 (학습 설정과 동일해야 함)
    NEW_TOKENS_PER_STEP = cfg.data.max_seq_len // 2  # 한 번에 생성할 길이 (CONTEXT_WINDOW의 절반 추천)

    # 4. 슬라이딩 윈도우 방식 긴 곡 생성

    current_ids = [composer_token_id, tokenizer["Bar_None"]]
    full_generated_sequence = list(current_ids)
    
    TARGET_LENGTH = cfg.target_length if hasattr(cfg, "target_length") and cfg.target_length and cfg.target_length <= 1024 * 16 else 1024 * 4  # 기본 4096
    CONTEXT_WINDOW = cfg.data.max_seq_len
    NEW_TOKENS_PER_STEP = CONTEXT_WINDOW // 2

    print(f">> 목표 길이: {TARGET_LENGTH} 토큰 (슬라이딩 방식)")
    pbar = tqdm(total=TARGET_LENGTH, desc="작곡 중")
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
                pad_token_id=0
            )

        # 3) 새로운 토큰만 추출해서 전체 시퀀스에 추가
        new_tokens = output[0, len(input_ids):].tolist()
        full_generated_sequence.extend(new_tokens)
        pbar.update(len(new_tokens))

    pbar.close()

    # 5. 저장
    print("\n>> 변환 및 저장 중...")
    final_midi_ids = [t for t in full_generated_sequence if t < len(tokenizer)]
    generated_midi = tokenizer.decode([final_midi_ids])
    save_path = f"output_{target_composer.replace(' ', '_')}_long.mid"
    generated_midi.dump_midi(save_path)
    print(f"\n=== 🎹 긴 곡 작곡 완료! 저장됨: {save_path} ===")

if __name__ == "__main__":
    main()