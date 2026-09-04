"""Evaluate a LoRA-fine-tuned FLAN-T5 adapter on SAMSum's test split.

Usage:
    python3 eval_model.py --model-dir outputs/flan-t5-samsum-lora/final
"""

import argparse

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import PeftModel

from common import MODEL_NAME, MAX_TARGET_LEN, build_compute_metrics, build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the saved LoRA adapter (e.g. .../final).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Which dataset split to evaluate on.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="If set, subsample the split to this many examples.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenized = build_dataset(tokenizer)
    eval_dataset = tokenized[args.split]
    if args.eval_samples is not None:
        eval_dataset = eval_dataset.shuffle(seed=42).select(
            range(min(args.eval_samples, len(eval_dataset)))
        )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, args.model_dir)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )

    eval_args = Seq2SeqTrainingArguments(
        output_dir="outputs/_eval_scratch",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    print(f"Evaluating {args.model_dir} on {len(eval_dataset)} '{args.split}' examples...")
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
