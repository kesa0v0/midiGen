import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def format_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_PROMPT.format(instruction=instruction, input=input_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3-8B with Unsloth QLoRA.")
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("train.jsonl"),
        help="Path to the Alpaca-style JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Trainer output directory.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("lora_adapter"),
        help="Directory to save the LoRA adapter.",
    )
    parser.add_argument(
        "--sanity-steps",
        type=int,
        default=0,
        help="Optional short run to sanity-check training before full epoch.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train_jsonl
    if not train_path.exists():
        raise FileNotFoundError(f"train.jsonl not found: {train_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(train_path), split="train")

    def format_example(example):
        instruction = (example.get("instruction") or "").strip()
        input_text = (example.get("input") or "").strip()
        output_text = (example.get("output") or "").strip()
        prompt = format_prompt(instruction, input_text)
        return {"text": prompt + output_text + tokenizer.eos_token}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    training_kwargs = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "fp16": False,
        "bf16": True,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "output_dir": str(args.output_dir),
        "logging_steps": 1,
        "save_steps": 250,
        "save_total_limit": 3,
    }
    if args.sanity_steps > 0:
        training_kwargs["max_steps"] = args.sanity_steps

    training_args = TrainingArguments(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer.train()

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter_dir))
    tokenizer.save_pretrained(str(args.adapter_dir))
    print(f"Saved LoRA adapter to {args.adapter_dir}")


if __name__ == "__main__":
    main()
