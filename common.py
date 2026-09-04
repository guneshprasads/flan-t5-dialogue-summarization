"""Shared dataset/model/metric building blocks for train.py and evaluate.py."""

import numpy as np
from datasets import load_dataset
import evaluate as hf_evaluate

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128
PROMPT_PREFIX = "Summarize the following conversation:\n\n"


def build_dataset(tokenizer):
    raw = load_dataset("knkarthick/samsum")

    def preprocess(batch):
        inputs = [PROMPT_PREFIX + d for d in batch["dialogue"]]
        model_inputs = tokenizer(
            inputs, max_length=MAX_INPUT_LEN, truncation=True
        )
        labels = tokenizer(
            text_target=batch["summary"], max_length=MAX_TARGET_LEN, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    return tokenized


def build_compute_metrics(tokenizer):
    rouge = hf_evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return {k: round(v * 100, 2) for k, v in result.items()}

    return compute_metrics
