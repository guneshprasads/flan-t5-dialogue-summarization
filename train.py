import argparse

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from common import MODEL_NAME, MAX_TARGET_LEN, build_compute_metrics, build_dataset

OUTPUT_DIR = "outputs/flan-t5-samsum-lora"


def build_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="If set, subsample the training set to this many examples.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="If set, subsample the validation set to this many examples.",
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = build_dataset(tokenizer)
    model = build_model()

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    if args.train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=42).select(
            range(min(args.train_samples, len(train_dataset)))
        )
    if args.eval_samples is not None:
        eval_dataset = eval_dataset.shuffle(seed=42).select(
            range(min(args.eval_samples, len(eval_dataset)))
        )

    print(
        f"Training on {len(train_dataset)} examples, "
        f"validating on {len(eval_dataset)} examples, for {args.epochs} epoch(s)."
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="none",
        fp16=False,  # MPS/CPU do not support fp16 training
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"\nSaved LoRA-adapted model to {args.output_dir}/final")
    print("Run evaluate.py against this directory to score it on the test split.")


if __name__ == "__main__":
    main()
