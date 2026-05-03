"""
tier1_finetune.py — Binary Sentinel Filter Fine-Tuning (Tier 1)

Fine-tunes distilbert-base-uncased on the DSD training split as a binary classifier:
    Negative (label 0): minimal — little to no depressive signal, safely filterable
    Positive (label 1): mild + moderate + severe — any clinically meaningful at-risk signal

Clinical rationale for the class split:
    Mild posts share surface-level language with minimal posts (low mood, some negativity)
    but represent a clinically meaningful category: early-onset depressive episodes that
    benefit from early intervention. Treating mild as negative would teach the filter to
    discard borderline posts — exactly the cases where the LLM's reasoning adds the most
    value. The filter's only job is "is any depressive signal present?"; mild clearly
    qualifies. Moderate and severe are unambiguously at-risk.

    Using the DSD training split (same distribution, same domain) rather than external
    or synthetic data avoids the domain-shift risk of training on generic "non-depressive"
    text that does not reflect how minimal-class social media posts actually read.

Class weighting:
    Per-class weights mirror Tier 2's SAMPLE_WEIGHTS for consistency across the pipeline:
        minimal: 1.0  |  mild: 2.5  |  moderate: 3.0  |  severe: 8.0
    The binary at_risk weight is the mean of the three at-risk multipliers (4.5),
    so the loss function applies the same severity bias as Tier 2's sampler.

Usage:
    python -m src.tier1_finetune
    Saves fine-tuned model to models/tier1_filter/

Run order (full pipeline):
    1. python -m src.tier1_finetune   (fast — CPU-capable, ~10 min on GPU)
    2. python -m src.tier2_finetune   (GPU required)
    3. python main.py
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from src.constants import ORDINAL_ORDER

BASE_MODEL_ID = "distilbert-base-uncased"

CONFIG = {
    "train_csv":  "data/train.csv",
    "output_dir": "models/tier1_filter",
    "epochs":     10, # A bunch of epoch just to run, will save the best one
    "max_length": 384,   # DSD posts are short; 256 covers >99% without padding waste
    "batch_size": 16,
    "lr":         1e-5,
}

# Positive class: any label with clinically meaningful depressive signal
AT_RISK_LABELS = {"mild", "moderate", "severe"}

# Per-class sampling weights — mirrors Tier 2's SAMPLE_WEIGHTS for consistency.
# The binary class weight for at_risk is the mean of the three at-risk multipliers,
# giving the loss function the same severity bias as Tier 2's sampler:
#   minimal: 1.0  |  mild: 2.5  |  moderate: 3.0  |  severe: 8.0
#   at_risk weight = mean(2.5, 3.0, 8.0) = 4.5
SAMPLE_WEIGHTS = {"minimal": 1.0, "mild": 2.5, "moderate": 3.0, "severe": 8.0}
AT_RISK_WEIGHT = sum(SAMPLE_WEIGHTS[l] for l in AT_RISK_LABELS) / len(AT_RISK_LABELS)  # 4.5


def to_binary_label(label: str) -> int:
    """Map DSD label to binary: 0 = minimal (negative), 1 = at-risk (positive)."""
    return 1 if label in AT_RISK_LABELS else 0


def compute_metrics(eval_pred) -> dict:
    """
    Report metrics that matter for a recall-priority gate:
      - at_risk_recall:  fraction of positive (at-risk) posts correctly passed through
      - precision:       fraction of passed posts that are truly at-risk (efficiency signal)
      - f1:              harmonic mean (general health check)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "at_risk_recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "precision":      f1_score(labels, preds, pos_label=1, average="binary", zero_division=0),
        "f1":             f1_score(labels, preds, average="binary", zero_division=0),
    }


def finetune() -> None:
    train_df = pd.read_csv(CONFIG["train_csv"])
    train_df = train_df[train_df["label"].isin(ORDINAL_ORDER)].reset_index(drop=True)

    train_df["binary_label"] = train_df["label"].map(to_binary_label)

    n_pos = int(train_df["binary_label"].sum())
    n_neg = len(train_df) - n_pos
    print(f"[Tier 1 Finetune] {len(train_df)} training examples loaded.")
    print(f"[Tier 1 Finetune] Negative (minimal): {n_neg} | Positive (at-risk): {n_pos}")
    print(f"[Tier 1 Finetune] Class weights — minimal: {SAMPLE_WEIGHTS['minimal']:.1f} | at_risk: {AT_RISK_WEIGHT:.1f}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    hf_dataset = Dataset.from_dict({
        "text":  train_df["text"].tolist(),
        "label": train_df["binary_label"].tolist(),
    }).map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=CONFIG["max_length"]),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=2,
        id2label={0: "minimal", 1: "at_risk"},
        label2id={"minimal": 0, "at_risk": 1},
    )

    # Weighted cross-entropy: penalises false negatives on the at-risk class
    # proportionally to the same severity bias used in Tier 2's sampler.
    class_weight = torch.tensor(
        [SAMPLE_WEIGHTS["minimal"], AT_RISK_WEIGHT], dtype=torch.float
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels,
                weight=class_weight.to(outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=CONFIG["output_dir"],
            num_train_epochs=CONFIG["epochs"],
            per_device_train_batch_size=CONFIG["batch_size"],
            per_device_eval_batch_size=CONFIG["batch_size"],
            learning_rate=CONFIG["lr"],
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="at_risk_recall",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
        ),
        train_dataset=hf_dataset,
        eval_dataset=hf_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    ).train()

    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"[Tier 1 Finetune] Model saved to '{CONFIG['output_dir']}'.")


if __name__ == "__main__":
    print("=" * 52 + "\n  Tier 1 Binary Filter Fine-Tuning\n" + "=" * 52)
    finetune()
