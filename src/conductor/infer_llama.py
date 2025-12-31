import argparse
from pathlib import Path

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

DEFAULT_INSTRUCTION = (
    "Generate a structured symbolic music representation based on the following metadata."
)


def build_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_PROMPT.format(instruction=instruction, input=input_text)


def extract_response(text: str) -> str:
    marker = "### Response:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a LoRA adapter.")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("lora_adapter"),
        help="Directory containing the LoRA adapter.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/llama-3-8b-bnb-4bit",
        help="Base model name.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction text for the prompt.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Genre: Pop, Key: C Major",
        help="Input metadata string for the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    FastLanguageModel.for_inference(model)

    prompt = build_prompt(args.instruction, args.input)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        use_cache=True,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(extract_response(decoded))


if __name__ == "__main__":
    main()
