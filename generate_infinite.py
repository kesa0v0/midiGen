import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import os
import logging
from src.model_module import MidiGenModule
from src.tokenizer_module import get_tokenizer
from tqdm import tqdm

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. 초기 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(cfg, "compile_model"): cfg.compile_model = False # 추론 시 컴파일 끄기

    # 2. 모델 & 토크나이저 로드
    tokenizer = get_tokenizer(cfg)
    ckpt_dir = Path("checkpoints") / cfg.project_name
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts: return
    
    log.info(f">> 모델 로드 중... {ckpts[-1]}")
    model_module = MidiGenModule(cfg, vocab_size=tokenizer.vocab_size)
    
    # 가중치 로드 및 FP16/BF16 변환 (VRAM 절약)
    checkpoint = torch.load(str(ckpts[-1]), map_location=device, weights_only=False)
    state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in checkpoint["state_dict"].items()}
    model_module.load_state_dict(state_dict, strict=False, assign=True)
    
    # 4070 Ti Super라면 bfloat16 추천
    model_module.to(dtype=torch.bfloat16).to(device).eval()

    # ==========================================
    # 🚀 무한 생성 루프 설정
    # ==========================================
    
    TOTAL_CHUNKS = 5       # 2048토큰 x 5번 = 약 10,000토큰 (원하는 만큼 늘리세요)
    CHUNK_LEN = 2048       # 한 번에 생성할 길이 (VRAM 안전하게 2048 추천)
    
    # 1) 시작 프롬프트 (작곡가 설정)
    current_ids = [tokenizer.start_token_id]
    if 'Global_Header' in tokenizer.tokens_structure:
        current_ids.append(tokenizer.tokens_structure['Global_Header'])
    
    target_composer = cfg.target_composer
    if hasattr(tokenizer, 'composer_map') and target_composer in tokenizer.composer_map:
        cid = tokenizer.composer_vocab_start + tokenizer.composer_map[target_composer]
        current_ids.append(cid)
        log.info(f">> 작곡가: {target_composer}")
    
    if 'Narrative_Stream' in tokenizer.tokens_structure:
        current_ids.append(tokenizer.tokens_structure['Narrative_Stream'])

    # 전체 곡을 담을 리스트
    full_song_ids = list(current_ids) 
    
    log.info(f"=== 🎹 무한 작곡 시작 (총 {TOTAL_CHUNKS} 조각 생성 예정) ===")

    for i in range(TOTAL_CHUNKS):
        torch.cuda.empty_cache() # 메모리 청소
        
        # 입력 준비: 전체 곡이 너무 길면, 최근 2048~4096개만 잘라서 힌트로 줌
        # Titans는 기억력이 좋아서 앞부분을 다시 읽으면 문맥을 복원함
        context_window = 4096 
        input_context = full_song_ids[-context_window:] 
        
        input_tensor = torch.tensor([input_context], device=device).long()
        
        log.info(f">> [Chunk {i+1}/{TOTAL_CHUNKS}] 생성 중... (입력 길이: {len(input_context)})")
        
        with torch.no_grad():
            # generate 함수는 입력+출력을 모두 반환함
            output = model_module.model.generate(
                input_tensor, 
                max_length=len(input_context) + CHUNK_LEN, # 입력 길이 + 새로 만들 길이
                temperature=1.0,
                top_k=40
            )
            
        # 새로 생긴 부분만 잘라내기
        new_tokens = output[0, len(input_context):].tolist()
        
        # 결과 합치기
        full_song_ids.extend(new_tokens)
        log.info(f"   -> {len(new_tokens)} 토큰 생성됨. (현재 총 길이: {len(full_song_ids)})")

        # 중간 저장 (혹시 꺼질까봐)
        if (i + 1) % 1 == 0:
            save_path = Path(f"generated_output/{cfg.project_name}/infinite_temp.mid")
            try:
                valid_tokens = [t for t in full_song_ids if t < tokenizer.vocab_size]
                tokenizer.decode(valid_tokens, save_path)
            except: pass

    # ==========================================
    # 💾 최종 저장
    # ==========================================
    final_path = Path(f"generated_output/{cfg.project_name}/titans_{target_composer.replace(' ', '_')}_full_length.mid")
    valid_tokens = [t for t in full_song_ids if t < tokenizer.vocab_size]
    
    log.info(f">> 최종 변환 중... (총 {len(valid_tokens)} 토큰)")
    try:
        tokenizer.decode(valid_tokens, final_path)
        log.info(f"=== 🎉 완성! 저장됨: {final_path} ===")
    except Exception as e:
        log.error(f"디코딩 실패: {e}")

if __name__ == "__main__":
    main()