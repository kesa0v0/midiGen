import hydra
from omegaconf import DictConfig
import torch
from miditok import REMI, TokenizerConfig
from pathlib import Path
import os
import json
from src.model_module import MidiGenModule # Lightning Module 불러오기

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

    # 3. 모델 로드 (Lightning의 마법!)
    # checkpoints 폴더에서 가장 최신 .ckpt 파일을 찾음
    ckpts = sorted(Path("checkpoints").glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        print("!! 체크포인트(.ckpt)가 없습니다. 먼저 train.py를 실행하세요.")
        return
    
    ckpt_path = str(ckpts[-1])
    print(f">> 로드 중: {ckpt_path}")

    # 모델 구조 + 가중치 자동 복구
    model_module = MidiGenModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model_module.to(device)
    model_module.eval()
    
    # 실제 GPT-2 모델 꺼내기
    model = model_module.model 

    # 4. 생성 시작
    # 시드: [작곡가, Bar]
    seed_ids = [composer_token_id, tokenizer["Bar_None"]]
    print(f">> 시드: {seed_ids}")

    print(">> 작곡 중...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=torch.tensor([seed_ids]).to(device),
            max_length=1024,
            do_sample=True,
            temperature=0.9,       # 창의성
            top_k=20,              # 안정성
            repetition_penalty=1.0,
            pad_token_id=0
        )

    # 5. 저장
    gen_token_ids = generated_ids[0].cpu().numpy().tolist()
    final_midi_ids = [t for t in gen_token_ids if t < len(tokenizer)]
    
    generated_midi = tokenizer.decode([final_midi_ids])
    save_path = f"output_{target_composer.replace(' ', '_')}.mid"
    generated_midi.dump_midi(save_path)
    print(f"\n=== 🎹 작곡 완료! 저장됨: {save_path} ===")

if __name__ == "__main__":
    main()