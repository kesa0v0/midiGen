import hydra
from omegaconf import DictConfig
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from miditok import REMI, TokenizerConfig
from pathlib import Path
import os

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Midigen V2 작곡 시작 (Device: {device}) ===")

    # 1. 토크나이저 로드
    tokenizer_config = TokenizerConfig(
        num_velocities=cfg.data.num_velocities, 
        use_chords=cfg.data.use_chords
    )
    tokenizer = REMI(tokenizer_config)
    print(f">> 토크나이저 로드 완료. Vocab Size: {len(tokenizer)}")

    # 2. 모델 로드
    model_config = GPT2Config(
        vocab_size=cfg.data.vocab_size,
        n_positions=cfg.data.max_seq_len,
        n_embd=cfg.model.dim,
        n_layer=cfg.model.depth,
        n_head=cfg.model.heads,
        pad_token_id=0
    )
    model = GPT2LMHeadModel(model_config).to(device)

    # 체크포인트 찾기
    ckpts = sorted(Path("checkpoints").glob("*.pt"), key=os.path.getmtime)
    if not ckpts:
        print("!! 체크포인트가 없습니다. 학습(train.py)을 먼저 하세요.")
        return
    ckpt_path = str(ckpts[-1])
    print(f">> 최신 체크포인트 로드: {ckpt_path}")
    
    # weights_only=True로 경고 끄기
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # 3. 시작 토큰 찾기 (Note On 토큰 찾기)
    # 0번(Padding)으로 시작하면 AI가 멍때립니다. 확실하게 "피아노 쳐!"(NoteOn) 신호를 줍니다.
    start_token_id = 2 # fallback
    for i in range(len(tokenizer)):
        token_str = tokenizer[i]
        if "NoteOn" in token_str:
            start_token_id = i
            print(f">> 시작 토큰 발견: {i} ({token_str})")
            break

    # 4. 생성 (Generation)
    print(">> 작곡 중... (반복 방지 적용)")
    generated_ids = model.generate(
        input_ids=torch.tensor([[start_token_id]]).to(device),
        max_length=512,
        do_sample=True,
        temperature=0.7,      # 0.6 ~ 0.8 추천
        top_k=30,
        repetition_penalty=1.2, # [중요] 반복하면 벌점 줌 (52, 52 방지)
        pad_token_id=0,
        eos_token_id=None
    )

    gen_token_ids = generated_ids[0].cpu().numpy().tolist()
    
    # 5. 결과 디버깅 (눈으로 확인)
    print(f"\nDEBUG: 생성된 토큰 앞부분 10개:")
    for tid in gen_token_ids[:10]:
        try:
            print(f"  {tid} -> {tokenizer[tid]}")
        except:
            print(f"  {tid} -> ???")

    # 6. 저장
    # 유효한 토큰만 필터링
    valid_ids = [t for t in gen_token_ids if t < len(tokenizer)]
    
    try:
        generated_midi = tokenizer.decode([valid_ids]) # 대괄호 [] 필수
        save_path = "output_song.mid"
        generated_midi.dump_midi(save_path)
        print(f"\n=== 작곡 완료! 저장됨: {save_path} ===")
    except Exception as e:
        print(f"변환 에러: {e}")

if __name__ == "__main__":
    main()