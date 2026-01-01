# commands

## conductor

### train

```bash
python src/conductor/train_llama.py \
    --data_path "data/processed/train.jsonl" \
    --base_dir "data/processed/llama/" \
    --num_train_epochs 1 \
    --save_steps 50 \
    --save_total_limit 10

python src/conductor/train_llama.py \
  --model_name data/processed/llama/lora_adapter_step_500 \
  --data_path data/processed/train.jsonl \
  --base_dir data/processed/llama \
  --max_steps 500 \
  --save_steps 50
```

### infer

```bash
python src/conductor/infer_llama.py \
  --adapter_dir data/processed/llama/lora_adapter \
  --base_dir data/processed/llama \
  --max_sections 8 \
  --bars_per_section 8 \
  --metadata "Genre: Pop, Key: C Major, Title: Demo_Title, Artist: Demo_Artist, BPM: 110"
```