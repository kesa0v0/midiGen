#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unsloth + Llama-3-8B (4bit QLoRA) SFT fine-tuning script.

Model: unsloth/llama-3-8b-bnb-4bit
GPU: RTX 4070 Ti Super (16GB VRAM) - bf16 권장

Outputs:
- Checkpoints & logs:  data/processed/llama/outputs/
- Final LoRA adapter:  data/processed/llama/lora_adapter/
- Inference test text: data/processed/llama/inference_test.txt

Dataset:
- JSONL: data/processed/llama/train.jsonl (기본값)
- Alpaca 템플릿으로 instruction/input/output -> 단일 text로 변환하여 SFTTrainer 학습
"""

import os
import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset

from unsloth import FastLanguageModel
# from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig


# -------------------------
# Alpaca prompt template
# -------------------------
ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_alpaca_example(example: Dict[str, Any], eos_token: str) -> Dict[str, str]:
    """
    train.jsonl 레코드를 Alpaca 템플릿에 맞춰 단일 문자열(text)로 변환.

    기대 필드(권장):
      - instruction: str
      - input: str (없거나 빈 문자열일 수 있음)
      - output: str

    호환을 위해 다음도 일부 허용:
      - prompt / response
      - text (이미 만들어진 학습 텍스트가 있는 경우)
    """
    if "text" in example and isinstance(example["text"], str) and example["text"].strip():
        # 이미 전처리된 text가 있으면 그대로 사용
        return {"text": example["text"].rstrip() + eos_token}

    instruction = (example.get("instruction") or example.get("prompt") or "").strip()
    _input = (example.get("input") or "").strip()
    output = (example.get("output") or example.get("response") or "").strip()

    # Alpaca 원형은 input이 비어있을 수 있으므로, 빈 경우에도 섹션을 유지
    formatted = ALPACA_TEMPLATE.format(
        instruction=instruction if instruction else "Follow the user request.",
        input=_input if _input else "",
        output=output
    ).rstrip()

    return {"text": formatted + eos_token}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/llama/train.jsonl",
                        help="Path to train.jsonl")
    parser.add_argument("--base_dir", type=str, default="data/processed/llama",
                        help="Base output directory")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-bnb-4bit",
                        help="Unsloth 4bit model repo")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Context length for training/inference")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="MVP test steps (나중에 늘리려면 값 변경)")
    # parser.add_argument("--max_steps", type=int, default=1000, help="(예) 본학습용으로 늘릴 때")

    parser.add_argument("--save_steps", type=int, default=25,
                        help="Intermediate checkpoint save frequency (steps)")
    parser.add_argument("--save_total_limit", type=int, default=4,
                        help="How many checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume from (optional)")

    args = parser.parse_args()

    base_dir = args.base_dir
    outputs_dir = os.path.join(base_dir, "outputs")
    adapter_dir = os.path.join(base_dir, "lora_adapter")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    # -------------------------
    # Load model (4bit) via Unsloth
    # -------------------------
    # RTX 40xx: bf16 권장
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # pad_token 세팅 (없으면 SFT에서 경고/에러 나는 경우가 있음)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # LoRA config (요청 사항 반영)
    # -------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,         # 일반적으로 QLoRA는 0.0을 많이 사용 (원하시면 변경)
        bias="none",
        use_gradient_checkpointing="unsloth",  # 요청사항
        random_state=42,
    )

    # -------------------------
    # Load dataset (JSONL) + format to Alpaca
    # -------------------------
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"train.jsonl not found: {args.data_path}")

    raw = load_dataset("json", data_files=args.data_path, split="train")

    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    train_ds = raw.map(
        lambda ex: format_alpaca_example(ex, eos_token),
        remove_columns=raw.column_names,
        desc="Formatting dataset with Alpaca template"
    )

    # -------------------------
    # Trainer (SFTTrainer) config
    # -------------------------
    sft_config = SFTConfig(
    output_dir=outputs_dir,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    warmup_steps=5,
    max_steps=args.max_steps,   # MVP: 100 (나중에 늘리기)

    learning_rate=2e-4,
    fp16=False,
    bf16=True,                 # RTX 40xx

    optim="adamw_8bit",
    weight_decay=0.01,

    logging_steps=1,

    # 중간 체크포인트
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,

    # Unsloth psutil 경로를 타지 않도록 명시 (중요)
    dataset_num_proc=1,

    # 기타
    report_to="none",
    seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=sft_config,
        packing=False,
    )

    # -------------------------
    # Train
    # -------------------------
    train_kwargs = {}
    if args.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint

    trainer.train(**train_kwargs)

    # -------------------------
    # Save LoRA adapter
    # -------------------------
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # -------------------------
    # Inference test (after training)
    # -------------------------
    FastLanguageModel.for_inference(model)

    # 임의 입력 예시 (요청: 'Genre: Pop, Key: C Major')
    test_instruction = "Given the genre and key, generate a short structured text output that follows the training style."
    test_input = "Genre: Pop, Key: C Major"

    test_prompt = ALPACA_TEMPLATE.format(
        instruction=test_instruction,
        input=test_input,
        output=""  # 생성은 Response 이후를 채우게 함
    )

    inputs = tokenizer([test_prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

    # 콘솔 출력
    print("\n================= INFERENCE TEST OUTPUT =================")
    print(decoded)
    print("=========================================================\n")

    # 파일 저장
    test_out_path = os.path.join(base_dir, "inference_test.txt")
    with open(test_out_path, "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    print(f"[OK] Saved adapter to: {adapter_dir}")
    print(f"[OK] Saved outputs/checkpoints to: {outputs_dir}")
    print(f"[OK] Saved inference test to: {test_out_path}")


if __name__ == "__main__":
    main()
