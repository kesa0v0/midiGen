import hydra
from omegaconf import DictConfig
import torch
from miditok import REMI, TokenizerConfig
from pathlib import Path
import os
import json
from src.models import MidigenTitans
from collections import Counter
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList

# [핵심] 단어장 크기를 벗어나는 생성을 막는 제한 장치
class RestrictVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, actual_vocab_size):
        self.actual_vocab_size = actual_vocab_size

    def __call__(self, input_ids, scores):
        # 단어장 크기(actual_vocab_size) 이상의 토큰 점수를 -Infinity로 만들어 선택 안 되게 함
        # 예: 모델은 10000개까지 알지만, 실제 단어는 5000개라면 5000번 이후는 절대 안 뽑음
        vocab_size = scores.shape[-1]
        if self.actual_vocab_size < vocab_size:
            scores[:, self.actual_vocab_size:] = -float('inf')
        return scores

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Midigen V3 [Final Generation] 작곡 시작 (Device: {device}) ===")

    # 1. 매핑 정보 로드
    if not os.path.exists("composer_map.json"):
        raise FileNotFoundError("composer_map.json이 없습니다.")
    
    with open("composer_map.json", "r") as f:
        mapping_info = json.load(f)
    
    composer_to_id = mapping_info["composer_to_id"]
    base_vocab_size = mapping_info["base_vocab_size"] # BPE 크기 (예: 5000)
    
    # 작곡가 선택
    target_composer = "Frédéric Chopin" 
    if target_composer not in composer_to_id:
        target_composer = list(composer_to_id.keys())[0]

    # 작곡가 토큰 ID 계산
    composer_token_id = base_vocab_size + composer_to_id[target_composer]
    print(f">> 선택된 작곡가: {target_composer} (Token ID: {composer_token_id})")

    # # 2. 토크나이저 로드 (tokenizer.json 필수)
    # if os.path.exists("tokenizer.json"):
    #     print(">> 학습된 BPE 토크나이저(tokenizer.json)를 로드합니다.")
    #     tokenizer = REMI(params="tokenizer.json")
    # else:
    #     print("!! 경고: tokenizer.json이 없습니다. 엉뚱한 음악이 나올 수 있습니다.")
    tokenizer_config = TokenizerConfig(
        num_velocities=cfg.data.num_velocities, 
        use_chords=cfg.data.use_chords,
        beat_res=eval(cfg.data.beat_res)
    )
    tokenizer = REMI(tokenizer_config)

    # 3. 모델 로드 (GPT-2)
    print(f">> Model Type: {cfg.model.type}")
    if cfg.model.type == "gpt2":
        model_config = GPT2Config(
            vocab_size=cfg.data.vocab_size, # config의 10000
            n_positions=cfg.data.max_seq_len,
            n_embd=cfg.model.dim,
            n_layer=cfg.model.depth,
            n_head=cfg.model.heads,
            pad_token_id=0
        )
        model = GPT2LMHeadModel(model_config).to(device)
    else:
        model = MidigenTitans(cfg).to(device)

    # 체크포인트 로드
    ckpts = sorted(Path("checkpoints").glob("*.pt"), key=os.path.getmtime)
    if not ckpts:
        print("!! 체크포인트 없음")
        return
    ckpt_path = str(ckpts[-1])
    print(f">> 로드 중: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # 4. 시드(Seed) 준비: [작곡가] + [Bar]
    # BPE 모델은 'NoteOn' 하나만 주면 헷갈려할 수 있으므로, 
    # 그냥 작곡가랑 시작 신호만 주고 알아서 하라고 하는 게 낫습니다.
    seed_ids = [composer_token_id, tokenizer["Bar_None"]] 
    print(f">> 시드 구성: [Composer: {target_composer}] + [Bar_None]")

    # 5. 생성 (LogitsProcessor 적용)
    actual_vocab_size = len(tokenizer)
    print(f">> 토크나이저 실제 크기: {actual_vocab_size} (이보다 큰 ID는 차단합니다)")
    
    # logits_processor = LogitsProcessorList([
    #     RestrictVocabLogitsProcessor(actual_vocab_size)
    # ])

    print(">> 작곡 중...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=torch.tensor([seed_ids]).to(device),
            max_length=1024,
            do_sample=True,
            temperature=0.8,       # 0.8: 너무 랜덤하지 않게 (Position 폭주 방지)
            top_k=20,              
            repetition_penalty=1.0, # [중요] 1.0 = 페널티 끔. 음악은 반복이 생명입니다.
            pad_token_id=0,
            # logits_processor=logits_processor # 제한 장치 장착
        )

    # 6. 저장 및 결과 분석
    gen_token_ids = generated_ids[0].cpu().numpy().tolist()

    # 통계 계산
    decoded_tokens = []
    valid_ids_for_midi = []
    
    # 작곡가 토큰(seed에 포함됨)은 통계에서 제외하고 생성된 것만 분석
    generated_part = gen_token_ids[len(seed_ids):]

    print(f">> 생성된 토큰 해독 중... (총 {len(generated_part)}개)")

    for tid in generated_part:
        # 1. 범위 체크
        if tid >= len(tokenizer):
            continue
            
        # 2. 존재 여부 체크 (KeyError 방지)
        try:
            # tokenizer[tid]가 실패하면 except로 넘어감
            token_str = tokenizer[tid] 
            decoded_tokens.append(token_str)
            valid_ids_for_midi.append(tid)
        except KeyError:
            # 325번 같은 유령 토큰은 무시
            continue
        except Exception as e:
            print(f"!! 토큰 해독 중 예외 발생 (ID: {tid}): {e}")
            continue
    
    counts = Counter([t.split('_')[0] for t in decoded_tokens if isinstance(t, str)])
    
    print("\n=== 생성 결과 통계 ===")
    print(f"총 생성 길이: {len(generated_part)}")
    print(f"유효한 토큰: {len(valid_ids_for_midi)}")
    print(f"🎵 음표(Pitch/NoteOn): {counts.get('Pitch', 0) + counts.get('NoteOn', 0)}")
    print(f"⏳ 시간(Position): {counts.get('Position', 0)}")
    print(f"🎹 화음/기타(Chord 등): {counts.get('Chord', 0)}")
    
    # MIDI 변환 (작곡가 토큰 제외하고 순수 음악 토큰만)
    # 시드에 있던 Bar_None은 포함해도 됨
    if "Bar_None" in tokenizer.vocab:
        start_token = tokenizer["Bar_None"]
    else:
        # 혹시 Bar_None도 없으면 0번이나 가장 자주 나오는 토큰으로 대체 (안전장치)
        start_token = valid_ids_for_midi[0] if valid_ids_for_midi else 0

    final_midi_ids = [start_token] + valid_ids_for_midi
    
    if len(valid_ids_for_midi) < 10:
        print("!! 경고: 생성된 음표가 너무 적습니다. (대부분이 유령 토큰이거나 생성 실패)")
    else:
        try:
            generated_midi = tokenizer.decode([final_midi_ids])
            save_path = f"output_{target_composer.replace(' ', '_')}.mid"
            generated_midi.dump_midi(save_path)
            print(f"\n=== 🎹 작곡 완료! 저장됨: {save_path} ===")
            print("이제 파일을 다운로드해서 들어보세요!")
        except Exception as e:
            print(f"MIDI 변환 에러: {e}")
            print("팁: 토크나이저 설정(Beat Resolution 등)이 학습 때와 다를 수도 있습니다.")

if __name__ == "__main__":
    main()