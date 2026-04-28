import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score

ORDINAL_ORDER = ["minimal", "mild", "moderate", "severe"]


def evaluate_tier1(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> float:
    """System Recall: % of 'severe' posts retained by Tier 1. Must stay high."""
    total = len(original_df[original_df["label"] == "severe"])
    if total == 0:
        return 100.0
    passed = len(filtered_df[filtered_df["label"] == "severe"])
    return (passed / total) * 100


def evaluate_tier2(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Returns (macro_precision, macro_f1, weighted_f1, ordinal_mae).

    Both macro and weighted F1 are reported:
      macro    — treats all classes equally; penalised by minority class failures
      weighted — reflects overall accuracy accounting for class imbalance (~72.8% minimal)
    Ordinal MAE penalises distant misclassifications (e.g. minimal↔severe) more than
    adjacent ones, consistent with the clinical motivation of the project.
    """
    valid = df[df["tier2_label"].isin(ORDINAL_ORDER) & df["label"].isin(ORDINAL_ORDER)].copy()

    if len(df) - len(valid):
        print(f"[Eval] Excluded {len(df) - len(valid)} rows with unparseable Tier 2 labels.")
    if valid.empty:
        return 0.0, 0.0, 0.0, float("nan")

    y_true, y_pred = valid["label"], valid["tier2_label"]

    order_map   = {l: i for i, l in enumerate(ORDINAL_ORDER)}
    ordinal_mae = float(np.mean(np.abs(y_true.map(order_map) - y_pred.map(order_map))))

    return (
        precision_score(y_true, y_pred, labels=ORDINAL_ORDER, average="macro",    zero_division=0),
        f1_score(       y_true, y_pred, labels=ORDINAL_ORDER, average="macro",    zero_division=0),
        f1_score(       y_true, y_pred, labels=ORDINAL_ORDER, average="weighted", zero_division=0),
        ordinal_mae,
    )


def print_classification_report(df: pd.DataFrame) -> None:
    valid = df[df["tier2_label"].isin(ORDINAL_ORDER) & df["label"].isin(ORDINAL_ORDER)]
    if valid.empty:
        return

    print("\nPer-Class Report (Tier 2):")
    print(classification_report(valid["label"], valid["tier2_label"], labels=ORDINAL_ORDER, zero_division=0))

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(valid["label"], valid["tier2_label"], labels=ORDINAL_ORDER)
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in ORDINAL_ORDER))
    for label, row in zip(ORDINAL_ORDER, cm):
        print(f"{label:>10}" + "".join(f"{v:>10}" for v in row))


def print_final_report(
    tier1_metrics:    dict,
    tier1_recall:     float,
    t2_precision:     float,
    t2_macro_f1:      float,
    t2_weighted_f1:   float,
    t2_ordinal_mae:   float,
) -> None:
    w = 52
    print("\n" + "=" * w)
    print("       CASCADE SYSTEM EVALUATION REPORT")
    print("=" * w)

    print("\n[Tier 1 — Sentinel Filter]")
    print(f"  Model              : mrm8488/distilroberta-base-finetuned-suicide-depression")
    print(f"  Threshold          : p > 0.30")
    print(f"  Posts In / Out     : {tier1_metrics['original_count']} → {tier1_metrics['passed_count']}")
    print(f"  Reduction          : {tier1_metrics['reduction_percentage']:.1f}% filtered out")
    print(f"  Latency per Post   : {tier1_metrics['latency_per_post_ms']:.2f} ms")
    print(f"  System Recall      : {tier1_recall:.1f}%  (severe cases retained)")

    print("\n[Tier 2 — Fine-tuned LLM Reasoning Engine]")
    print(f"  Base Model         : meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"  Adaptation         : QLoRA (4-bit NF4, rank=16) + WeightedRandomSampler")
    print(f"  Moderation Prec.   : {t2_precision:.4f}  (macro)")
    print(f"  F1 — Macro         : {t2_macro_f1:.4f}")
    print(f"  F1 — Weighted      : {t2_weighted_f1:.4f}")
    print(f"  Ordinal MAE        : {t2_ordinal_mae:.4f}  (avg severity-level distance)")

    print("\n" + "=" * w + "\n")
    