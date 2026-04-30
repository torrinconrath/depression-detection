"""
tier1_finetune.py — Fine-tune a 4-class severity classifier for Tier 1.

Replaces the binary DistilRoBERTa filter with a model that understands all four
severity labels, allowing Tier 1 to filter out true minimal posts rather than
approximating it with a binary depression threshold.

Model: distilroberta-base (fast, GPU-accelerated, ~300MB)
Task:  sequence classification — minimal / mild / moderate / severe

Usage:
    python -m src.tier1_finetune
"""

import os
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from src.constants import ORDINAL_ORDER

CONFIG = {
    "train_csv":   "data/train.csv",
    "output_dir":  "models/tier1_classifier",
    "base_model":  "distilroberta-base",
    "max_length":  384,
    "epochs":      4,
    "batch_size":  32,
    "lr":          2e-5,
    # Severe is boosted most — missing a severe at Tier 1 means Tier 2 never
    # sees it. Moderate and mild are also boosted to counter the 72.8% minimal prior.
    "sample_weights": {"minimal": 1.0, "mild": 2.5, "moderate": 3.0, "severe": 8.0},
}

LABEL2ID = {l: i for i, l in enumerate(ORDINAL_ORDER)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class DSDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(df["label"].map(LABEL2ID).tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]


def build_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    weights = df["label"].map(CONFIG["sample_weights"]).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=len(df), replacement=True)


def evaluate(model, loader, device) -> tuple[float, float, list, list]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds   = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    severe_f1 = f1_score(
        all_labels, all_preds,
        labels=[LABEL2ID["severe"]], average="macro", zero_division=0
    )
    return macro_f1, severe_f1, all_labels, all_preds


def finetune() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Tier1-Finetune] Running on {str(device).upper()}")

    train_df = pd.read_csv(CONFIG["train_csv"])
    train_df = train_df[train_df["label"].isin(ORDINAL_ORDER)].reset_index(drop=True)
    print(f"[Tier1-Finetune] {len(train_df)} training examples loaded.")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    model     = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"],
        num_labels=len(ORDINAL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_dataset = DSDDataset(train_df, tokenizer, CONFIG["max_length"])
    train_loader  = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=build_sampler(train_df),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    best_severe_f1 = 0.0
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch, labels in train_loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            loss   = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        macro_f1, severe_f1, all_labels, all_preds = evaluate(model, train_loader, device)
        print(
            f"[Tier1-Finetune] Epoch {epoch}/{CONFIG['epochs']} | "
            f"loss: {avg_loss:.4f} | macro F1: {macro_f1:.4f} | severe F1: {severe_f1:.4f}"
        )

        # Save best checkpoint by severe F1 — the clinical priority
        if severe_f1 > best_severe_f1:
            best_severe_f1 = severe_f1
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            model.save_pretrained(CONFIG["output_dir"])
            tokenizer.save_pretrained(CONFIG["output_dir"])
            print(f"[Tier1-Finetune] ✓ New best severe F1 ({severe_f1:.4f}) — checkpoint saved.")

    print(f"\n[Tier1-Finetune] Training complete. Best severe F1: {best_severe_f1:.4f}")
    print(f"[Tier1-Finetune] Classifier saved to '{CONFIG['output_dir']}'.")

    # Final report on training set
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["output_dir"]).to(device)
    final_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"])
    _, _, all_labels, all_preds = evaluate(model, final_loader, device)
    label_names = [ID2LABEL[i] for i in range(len(ORDINAL_ORDER))]
    print("\n[Tier1-Finetune] Classification report (train set — sanity check):")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))


if __name__ == "__main__":
    print("=" * 52 + "\n  Tier 1 Classifier Finetuning\n" + "=" * 52)
    finetune()
    