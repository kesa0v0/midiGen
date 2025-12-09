import hydra
from omegaconf import DictConfig
import torch
from miditok import REMI, TokenizerConfig
from pathlib import Path
import os
import json
from src.models import MidigenTitans
from collections import Counter

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Midigen V2 [Composer Control] 작곡 시작 (Device: {device}) ===")

    # 1. 매핑 정보 로드 (Composer ID 알기 위해)
    if not os.path.exists("composer_map.json"):
        raise FileNotFoundError("composer_map.json이 없습니다. preprocess.py를 먼저 실행했나요?")
    
    with open("composer_map.json", "r") as f:
        mapping_info = json.load(f)
    
    composer_to_id = mapping_info["composer_to_id"]
    base_vocab_size = mapping_info["base_vocab_size"]
    
    # [설정] 원하는 작곡가 이름을 여기에 적으세요! (JSON 파일 참고)
    # 예: "Frédéric Chopin", "Ludwig van Beethoven", "Johann Sebastian Bach"
    target_composer = "Frédéric Chopin" 
    
    if target_composer not in composer_to_id:
        print(f"!! 경고: '{target_composer}'는 목록에 없습니다. 랜덤으로 아무나 고릅니다.")
        target_composer = list(composer_to_id.keys())[0]

    composer_token_id = base_vocab_size + composer_to_id[target_composer]
    print(f">> 선택된 작곡가: {target_composer} (Token ID: {composer_token_id})")

    # 2. 토크나이저 로드 (Resolution 설정 적용)
    # config.yaml의 beat_res 문자열을 딕셔너리로 변환
    beat_res_dict = eval(cfg.data.beat_res)
    
    tokenizer_config = TokenizerConfig(
        num_velocities=cfg.data.num_velocities, 
        use_chords=cfg.data.use_chords,
        beat_res=beat_res_dict # [중요] V2에서 바뀐 해상도 적용
    )
    tokenizer = REMI(tokenizer_config)

    # 3. 모델 로드
    model = MidigenTitans(cfg).to(device)

    # 체크포인트 로드 (Loss 0.7짜리 괴물 로드)
    ckpts = sorted(Path("checkpoints").glob("*.pt"), key=os.path.getmtime)
    if not ckpts:
        print("!! 체크포인트 없음")
        return
    ckpt_path = str(ckpts[-1])
    print(f">> 로드 중: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # 4. 시드(Seed) 준비: [작곡가 토큰] + [도미솔도 멜로디]
    print(">> 시드 멜로디 생성 중...")
    
    # (1) 작곡가 토큰
    seed_ids = [composer_token_id] 
    
    # (2) 멜로디 (도-미-솔-도)
    def get_token(prefix, default_idx=0):
        # prefix로 시작하는 토큰 중 하나 찾기
        candidates = [t for t in tokenizer.vocab if t.startswith(prefix)]
        if candidates:
            # 적당히 중간값 혹은 정렬 후 선택
            candidates.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
            return tokenizer[candidates[len(candidates)//2]]
        return tokenizer["Bar_None"] # Fallback

    try:
        # V2는 해상도가 달라서 토큰 이름이 다를 수 있으므로 동적 검색
        # Position, Duration 등은 토크나이저 vocab에서 검색해서 구성
        
        # 간략화된 시드 주입 (오류 방지 위해 단순화)
        # 작곡가 토큰만 줘도 스타일이 나옵니다. 여기서는 작곡가 토큰 + 시작(Bar)만 줍니다.
        # 시드 멜로디까지 넣으면 좋지만, Resolution 변경으로 토큰 이름 맞추기 까다로울 수 있음.
        
        # 그래도 도미솔도는 넣어봅시다 (동적 검색)
        pitch_60 = tokenizer["Pitch_60"] if "Pitch_60" in tokenizer.vocab else tokenizer["NoteOn_60"]
        
        # 시드에 추가
        seed_ids.append(tokenizer["Bar_None"])
        seed_ids.append(pitch_60) # 첫 음 '도' 하나만 줘서 시작 유도
        
        print(f">> 시드 구성 완료: [Composer: {target_composer}] + [Bar] + [NoteOn_60]")

    except Exception as e:
        print(f"!! 시드 구성 중 에러 (기본으로 진행): {e}")

    # 5. 생성
    print(">> 작곡 중...")

    print(f"DEBUG: 시드 토큰: {seed_ids}")
    print(f"DEBUG: 가장 큰 토큰 ID: {max(seed_ids)}")
    print(f"DEBUG: 모델이 아는 단어장 크기(Embedding Size): {model.token_emb.num_embeddings}")
    
    if max(seed_ids) >= model.token_emb.num_embeddings:
        print("!! [치명적 오류] 입력 토큰이 모델의 단어장보다 큽니다! Config의 vocab_size를 늘려야 합니다.")
        exit()
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=torch.tensor([seed_ids]).to(device),
            max_length=1024,       # 길게 뽑아봅시다
            temperature=0.95,       # 자신감이 있으니 0.8 정도
            top_k=50,
            repetition_penalty=1.0 # 패널티 거의 끔 (자연스러운 반복 허용)
        )

    # 6. 저장
    gen_token_ids = generated_ids[0].cpu().numpy().tolist()

    # 토큰 통계 출력
    # 토큰 ID를 사람이 읽을 수 있는 텍스트로 변환
    decoded_tokens = [tokenizer[tid] for tid in gen_token_ids if tid < len(tokenizer)]
    counts = Counter([t.split('_')[0] for t in decoded_tokens]) # Prefix만 셈 (Pitch, Position...)
    
    print("\n=== 토큰 통계 (이 비율이 중요함) ===")
    print(f"총 토큰 수: {len(decoded_tokens)}")
    print(f"🎵 음표(Pitch/NoteOn): {counts.get('Pitch', 0) + counts.get('NoteOn', 0)}개")
    print(f"⏳ 시간이동(Position): {counts.get('Position', 0)}개")
    print(f"📏 지속시간(Duration): {counts.get('Duration', 0)}개")
    
    if (counts.get('Pitch', 0) + counts.get('NoteOn', 0)) < 100:
        print("!! 경고: 음표가 너무 적습니다! 여전히 쉼표만 찍고 있습니다.")
    else:
        print(">> 상태 양호: 음표가 충분히 생성되었습니다.")
    
    # 작곡가 토큰(300번대)은 MIDI 변환 시 에러나므로 제거해야 함!
    # 기본 vocab size보다 큰 ID는 필터링
    valid_ids = [t for t in gen_token_ids if t < mapping_info["base_vocab_size"]]
    
    try:
        generated_midi = tokenizer.decode([valid_ids])
        save_path = f"output_{target_composer.replace(' ', '_')}.mid"
        generated_midi.dump_midi(save_path)
        print(f"\n=== 작곡 완료! 저장됨: {save_path} ===")
        
    except Exception as e:
        print(f"변환 에러: {e}")
        # 디버깅용: Composer 토큰이 섞여 들어갔는지 확인
        print(f"Max Token ID: {max(gen_token_ids)}")

if __name__ == "__main__":
    main()