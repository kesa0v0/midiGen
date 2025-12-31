#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference-only script for Unsloth + Llama-3-8B (4bit) with a trained LoRA adapter.

Default paths (per your pipeline):
- Adapter dir: data/processed/llama/lora_adapter
- Output dir : data/processed/llama/inference

Unsloth docs: You can load your finetuned LoRA directory directly via FastLanguageModel.from_pretrained(model_name="lora_model"). :contentReference[oaicite:1]{index=1}
"""

import os
import re
import argparse
from typing import Optional, List, Tuple

import torch
from unsloth import FastLanguageModel

# Same Alpaca template used in training
ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def build_instruction(require_sections: bool = True,
                      max_sections: Optional[int] = None,
                      bars_per_section: Optional[int] = None) -> str:
    """
    강화된 지시문:
      - 반드시 [GLOBAL], [INSTRUMENTS], [FORM], [SECTION:*] 포함
      - FORM에 등장하는 섹션들에 대해 SECTION 블록을 생성하도록 강제
      - 과도한 길이/환각 억제를 위해 max_sections / bars_per_section 옵션 제공
    """
    parts = [
        "Generate a structured symbolic music representation based on the following metadata.",
        "The output MUST strictly follow this format and ordering:",
        "1) [GLOBAL] 2) [INSTRUMENTS] 3) [FORM] 4) [SECTION:*] blocks for every section listed in [FORM].",
        "In [FORM], use a reasonable number of sections and avoid meaningless repetition.",
        "For each [SECTION:*], include at least: BARS, PROG, CTRL (even if PROG is empty, keep the field).",
        "Use concise but complete sections. Do not stop after [FORM]."
    ]

    if require_sections:
        parts.append("IMPORTANT: You MUST include [SECTION:*] blocks. Do NOT end after [FORM].")

    if max_sections is not None:
        parts.append(f"Limit the total number of unique sections in [FORM] to at most {max_sections}.")

    if bars_per_section is not None:
        parts.append(f"Use BARS={bars_per_section} for each section unless metadata strongly suggests otherwise.")

    return " ".join(parts)

def build_form_only_instruction(max_sections: Optional[int] = None) -> str:
    parts = [
        "Generate a structured symbolic music representation based on the following metadata.",
        "Output ONLY the following sections in order: [GLOBAL], [INSTRUMENTS], [FORM].",
        "Do NOT include any [SECTION:*] blocks.",
        "In [FORM], list sections with bar counts in parentheses (e.g., A1(8) > B1(8))."
    ]
    if max_sections is not None:
        parts.append(f"Limit the total number of unique sections in [FORM] to at most {max_sections}.")
    parts.append("Stop after finishing [FORM].")
    return " ".join(parts)

def build_sections_only_instruction(section_list: List[str],
                                    bars_per_section: Optional[int] = None) -> str:
    sections_str = ", ".join(section_list)
    parts = [
        "Generate ONLY the [SECTION:*] blocks for the following section list, in order:",
        sections_str,
        "Do NOT include [GLOBAL], [INSTRUMENTS], or [FORM].",
        "For each [SECTION:*], include at least: BARS, PROG, CTRL (even if PROG is empty, keep the field)."
    ]
    if bars_per_section is not None:
        parts.append(f"Use BARS={bars_per_section} for each section unless metadata strongly suggests otherwise.")
    return " ".join(parts)

def split_prompt_response(decoded: str) -> Tuple[str, str]:
    marker = "### Response:"
    idx = decoded.find(marker)
    if idx == -1:
        return "", decoded
    prefix = decoded[:idx + len(marker)]
    response = decoded[idx + len(marker):].lstrip("\n")
    return prefix, response

def extract_form_block(text: str) -> Optional[str]:
    match = re.search(r"\[FORM\]\s*(.*?)(?=\n\[|$)", text, flags=re.S)
    if not match:
        return None
    return match.group(1).strip()

def extract_section_names(form_block: str, max_sections: Optional[int]) -> List[str]:
    if not form_block:
        return []
    candidates = re.findall(r"([A-Z][A-Z0-9_]*)\s*\(", form_block)
    unique = []
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        unique.append(name)
        seen.add(name)
        if max_sections is not None and len(unique) >= max_sections:
            break
    return unique

def trim_after_form(response: str) -> str:
    match = re.search(r"\[FORM\].*?(?=\n\[|$)", response, flags=re.S)
    if not match:
        return response.strip()
    return response[:match.end()].rstrip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="data/processed/llama",
                        help="Base directory for outputs (must be under data/processed/llama)")
    parser.add_argument("--adapter_dir", type=str, default="data/processed/llama/lora_adapter",
                        help="Directory where LoRA adapter was saved (model.save_pretrained).")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Keep 4bit loading for VRAM efficiency (default True).")
    parser.add_argument("--local_files_only", action="store_true", default=False,
                        help="If set, avoids downloading from HF and uses local cache only.")

    # Generation controls
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Increase to allow full [SECTION:*] blocks.")
    parser.add_argument("--max_new_tokens_form", type=int, default=512,
                        help="Token budget for pass-1 [FORM]-only generation.")
    parser.add_argument("--min_new_tokens", type=int, default=800,
                        help="Prevents early stopping before sections appear.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.12)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)

    # Structure constraints (optional but recommended to prevent runaway FORM)
    parser.add_argument("--max_sections", type=int, default=8,
                        help="Caps section count in [FORM] (helps prevent A1~Y1 runaway).")
    parser.add_argument("--bars_per_section", type=int, default=8,
                        help="Forces consistent section length (typical in your data).")

    # Input metadata
    parser.add_argument("--metadata", type=str, default="Genre: Pop, Key: C Major, BPM: 110",
                        help='Metadata string, e.g. "Genre: Pop, Key: C Major, Title: X, Artist: Y, BPM: 110"')

    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = os.path.join(base_dir, "inference")
    os.makedirs(out_dir, exist_ok=True)

    # Load finetuned LoRA directory directly
    # Unsloth inference doc shows loading YOUR trained model directory via FastLanguageModel.from_pretrained(model_name="lora_model") :contentReference[oaicite:2]{index=2}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,          # RTX 40xx
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    # Pass 1: [GLOBAL], [INSTRUMENTS], [FORM] only
    instruction_form = build_form_only_instruction(max_sections=args.max_sections)
    prompt_form = ALPACA_TEMPLATE.format(
        instruction=instruction_form,
        input=args.metadata
    )
    inputs_form = tokenizer([prompt_form], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_form = model.generate(
            **inputs_form,
            max_new_tokens=args.max_new_tokens_form,
            min_new_tokens=0,
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded_form = tokenizer.decode(generated_form[0], skip_special_tokens=True)
    prefix, response_form = split_prompt_response(decoded_form)
    form_block = extract_form_block(response_form or decoded_form)
    section_names = extract_section_names(form_block or "", args.max_sections)

    if section_names:
        print(f"[INFO] Parsed {len(section_names)} section(s) from [FORM]: {', '.join(section_names)}")

        # Pass 2: [SECTION:*] only
        instruction_sections = build_sections_only_instruction(
            section_list=section_names,
            bars_per_section=args.bars_per_section
        )
        prompt_sections = ALPACA_TEMPLATE.format(
            instruction=instruction_sections,
            input=args.metadata
        )
        inputs_sections = tokenizer([prompt_sections], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_sections = model.generate(
                **inputs_sections,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded_sections = tokenizer.decode(generated_sections[0], skip_special_tokens=True)
        _, response_sections = split_prompt_response(decoded_sections)

        response_form_trimmed = trim_after_form(response_form or decoded_form)
        combined_response = response_form_trimmed.rstrip() + "\n\n" + response_sections.strip() + "\n"
        if prefix:
            decoded = prefix.rstrip() + "\n" + combined_response
        else:
            decoded = combined_response
    else:
        print("[WARN] No sections parsed from [FORM]; skipping pass-2 generation.")
        decoded = decoded_form

    # Save outputs
    out_path = os.path.join(out_dir, "inference_output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    print("\n================= INFERENCE OUTPUT =================")
    print(decoded)
    print("====================================================\n")
    print(f"[OK] Saved inference output to: {out_path}")

if __name__ == "__main__":
    main()
