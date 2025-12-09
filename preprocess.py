import hydra
from omegaconf import DictConfig
import pandas as pd
import os
import json
from datasets import Dataset
from miditok import REMI, TokenizerConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"=== Midigen V2 전처리 시작: Composer Tagging ===")
    
    # 1. 메타데이터 로드
    df = pd.read_csv(cfg.data.csv_path)
    df['full_path'] = df['midi_filename'].apply(lambda x: os.path.join(cfg.data.raw_path, x))
    
    # 2. 작곡가 ID 매핑 생성 (Composer Mapping)
    # canonical_composer 컬럼을 사용 (철자가 통일되어 있음)
    composers = sorted(df['canonical_composer'].unique().tolist())
    composer_to_id = {name: i for i, name in enumerate(composers)}
    
    print(f">> 작곡가 {len(composers)}명 발견.")
    print(f">> 예시: {list(composer_to_id.items())[:3]}...")

    # 3. 토크나이저 설정 (Beat Resolution 적용)
    beat_res_dict = eval(cfg.data.beat_res)
    
    tokenizer_config = TokenizerConfig(
        num_velocities=cfg.data.num_velocities, 
        use_chords=cfg.data.use_chords,
        beat_res=beat_res_dict
    )
    tokenizer = REMI(tokenizer_config)
    
    base_vocab_size = len(tokenizer)
    print(f">> 기본 MIDI Vocab Size: {base_vocab_size}")
    
    final_vocab_size = base_vocab_size + len(composers)
    print(f">> 최종 Vocab Size (작곡가 포함): {final_vocab_size}")
    print(f"!! [중요] config.yaml의 vocab_size를 {final_vocab_size} (여유있게 {final_vocab_size + 10})로 수정하세요 !!")

    # [수정] JSON 저장을 위해 튜플 키 (0, 4)를 문자열 "(0, 4)"로 변환
    beat_res_serializable = {str(k): v for k, v in beat_res_dict.items()}

    mapping_info = {
        "composer_to_id": composer_to_id,
        "base_vocab_size": base_vocab_size,
        "beat_res": beat_res_serializable # [수정] 변환된 딕셔너리 저장
    }
    
    with open("composer_map.json", "w") as f:
        json.dump(mapping_info, f, indent=4)
    print(">> composer_map.json 저장 완료")

    # 4. 변환 함수 (Composer Token Injection)
    def tokenize_midi_with_composer(examples):
        token_ids_list = []
        for path, composer_name in zip(examples['full_path'], examples['canonical_composer']):
            try:
                # MIDI 토큰화
                tokens = tokenizer(path)
                ids = tokens[0].ids[:cfg.data.max_seq_len - 1] # 작곡가 토큰 자리 1개 남김
                
                # [핵심] 작곡가 토큰을 맨 앞에 추가
                # 작곡가 ID = base_vocab_size + 고유번호
                comp_token = base_vocab_size + composer_to_id[composer_name]
                ids.insert(0, comp_token)
                
                token_ids_list.append(ids)
            except Exception as e:
                # print(f"Error: {path}") # 로그 너무 많으면 주석 처리
                token_ids_list.append([])
        return {'input_ids': token_ids_list}

    # 5. 실행
    hf_dataset = Dataset.from_pandas(df[['split', 'full_path', 'canonical_composer']])
    
    tokenized_datasets = hf_dataset.map(
        tokenize_midi_with_composer, 
        batched=True, 
        batch_size=32, 
        num_proc=4,
        remove_columns=['full_path', 'canonical_composer']
    )

    tokenized_datasets.save_to_disk(cfg.data.processed_path)
    print("=== V2 전처리 완료 ===")

if __name__ == "__main__":
    main()