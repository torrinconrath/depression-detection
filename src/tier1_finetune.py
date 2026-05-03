"""
tier1_finetune.py — Sentinel Filter Fine-Tuning (Tier 1)

Fine-tunes distilbert-base-uncased on the DSD training split.
Unlike a strict binary model, this trains on all 4 severity classes (minimal, 
mild, moderate, severe) to preserve distinct linguistic profiles and apply 
strict individual loss penalties. 

Class weighting:
    Per-class weights mirror Tier 2's SAMPLE_WEIGHTS for consistency across the pipeline:
        minimal: 1.0  |  mild: 2.5  |  moderate: 3.0  |  severe: 8.0
    This ensures the model is severely penalised (8.0x) if it predicts 'minimal' 
    for a 'severe' post.

Usage:
    python -m src.tier1_finetune
    Saves fine-tuned model to models/tier1_filter/
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
    "epochs":     4,       
    "max_length": 384,    
    "batch_size": 16,
    "lr":         2e-5,
}

# Per-class sampling weights — mirrors Tier 2's SAMPLE_WEIGHTS for consistency.
SAMPLE_WEIGHTS = {"minimal": 1.0, "mild": 2.5, "moderate": 3.0, "severe": 8.0}
LABEL2ID = {"minimal": 0, "mild": 1, "moderate": 2, "severe": 3}


def compute_metrics(eval_pred) -> dict:
    """
    Report metrics that matter for a recall-priority gate.
    Squashes 4-class predictions back into binary (minimal=0 vs at_risk>0) 
    solely for the purpose of monitoring tracking metrics during training.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Binarize for evaluation metrics: 0 stays 0, (1, 2, 3) become 1
    binary_preds = (preds > 0).astype(int)
    binary_labels = (labels > 0).astype(int)
    
    return {
        "at_risk_recall": recall_score(binary_labels, binary_preds, pos_label=1, zero_division=0),
        "precision":      f1_score(binary_labels, binary_preds, pos_label=1, average="binary", zero_division=0),
        "f1":             f1_score(binary_labels, binary_preds, average="binary", zero_division=0),
    }


def finetune() -> None:
    train_df = pd.read_csv(CONFIG["train_csv"])
    train_df = train_df[train_df["label"].isin(ORDINAL_ORDER)].reset_index(drop=True)

    # Map labels to 0, 1, 2, 3
    train_df["label_id"] = train_df["label"].map(LABEL2ID)

    print(f"[Tier 1 Finetune] {len(train_df)} training examples loaded.")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    hf_dataset = Dataset.from_dict({
        "text":  train_df["text"].tolist(),
        "label": train_df["label_id"].tolist(),
    }).map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=CONFIG["max_length"]),
        batched=True,
    )

    # Set up the model for 4 classes
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=4,
        id2label={0: "minimal", 1: "mild", 2: "moderate", 3: "severe"},
        label2id=LABEL2ID,
    )

    # Apply the exact weights without averaging them
    class_weight = torch.tensor(
        [SAMPLE_WEIGHTS["minimal"], SAMPLE_WEIGHTS["mild"], 
         SAMPLE_WEIGHTS["moderate"], SAMPLE_WEIGHTS["severe"]], 
        dtype=torch.float
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
