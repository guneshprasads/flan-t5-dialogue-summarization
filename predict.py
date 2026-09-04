"""Generate summaries with a LoRA-fine-tuned FLAN-T5 adapter.

Examples:
    # Summarize a few dialogues from the test split, with reference summaries
    python3 predict.py --model-dir outputs/flan-t5-small-samsum-lora-quick/final

    # Compare the fine-tuned model against the untuned base model
    python3 predict.py --model-dir outputs/... --compare-base

    # Summarize your own dialogue
    python3 predict.py --model-dir outputs/... --dialogue "Amy: hey, lunch? \\nBen: sure, 1pm"
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from common import MODEL_NAME, MAX_INPUT_LEN, MAX_TARGET_LEN, PROMPT_PREFIX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the saved LoRA adapter (e.g. .../final).",
    )
    parser.add_argument(
        "--dialogue",
        type=str,
        default=None,
        help="Summarize this dialogue instead of samples from the test split.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many test-split dialogues to summarize.",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also show what the untuned base model produces, for comparison.",
    )
    return parser.parse_args()


def summarize(model, tokenizer, dialogue):
    inputs = tokenizer(
        PROMPT_PREFIX + dialogue,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_TARGET_LEN, num_beams=4
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()

    base_only = None
    if args.compare_base:
        base_only = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        base_only.eval()

    if args.dialogue is not None:
        samples = [{"dialogue": args.dialogue, "summary": None}]
    else:
        test_split = load_dataset("knkarthick/samsum")["test"]
        samples = test_split.shuffle(seed=42).select(range(args.num_samples))

    for i, sample in enumerate(samples, start=1):
        print("=" * 70)
        print(f"DIALOGUE {i}\n{sample['dialogue']}\n")
        if sample["summary"]:
            print(f"REFERENCE SUMMARY:\n  {sample['summary']}\n")
        if base_only is not None:
            print(f"BASE MODEL (no fine-tuning):\n  {summarize(base_only, tokenizer, sample['dialogue'])}\n")
        print(f"FINE-TUNED MODEL:\n  {summarize(model, tokenizer, sample['dialogue'])}\n")


if __name__ == "__main__":
    main()
