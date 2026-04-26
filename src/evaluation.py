import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)

ORDINAL_ORDER = ["minimal", "mild", "moderate", "severe"]


# ── Tier 1 ─────────────────────────────────────────────────────────────────────

def evaluate_tier1(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> float:
    """
    System Recall: percentage of 'severe' posts that passed the Tier 1 filter.
    Missing a severe case is the worst possible outcome — this must stay high.

    Returns:
        system_recall: float in [0, 100]
    """
    total_severe = len(original_df[original_df["label"] == "severe"])
    if total_severe == 0:
        print("[Eval] Warning: No 'severe' labels in dataset. Recall set to 100%.")
        return 100.0

    passed_severe = len(filtered_df[filtered_df["label"] == "severe"])
    return (passed_severe / total_severe) * 100


# ── Tier 2 ─────────────────────────────────────────────────────────────────────

def mean_absolute_error_ordinal(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Ordinal MAE: average severity-level distance between true and predicted labels.
    e.g. predicting 'mild' for a 'severe' case = distance of 2, not 1.
    This is clinically more meaningful than flat accuracy given the ordinal scale.
    """
    order_map = {label: i for i, label in enumerate(ORDINAL_ORDER)}
    true_idx = y_true.map(order_map)
    pred_idx = y_pred.map(order_map)
    return float(np.mean(np.abs(true_idx - pred_idx)))


def evaluate_tier2(filtered_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Evaluates Tier 2 LLM outputs against ground truth labels.

    Returns:
        (macro_precision, macro_f1, weighted_f1, ordinal_mae)

    Reports both macro and weighted F1:
      - Macro F1: treats all classes equally — penalised by minority class failures.
      - Weighted F1: accounts for class imbalance — reflects overall system accuracy.
    Both are reported because the dataset is heavily imbalanced (~72.8% minimal).
    """
    valid_df = filtered_df[
        filtered_df["tier2_label"].isin(ORDINAL_ORDER) &
        filtered_df["label"].isin(ORDINAL_ORDER)
    ].copy()

    invalid_count = len(filtered_df) - len(valid_df)
    if invalid_count > 0:
        print(f"[Eval] Excluded {invalid_count} rows with unparseable Tier 2 labels.")

    if len(valid_df) == 0:
        print("[Eval] Warning: No valid predictions to evaluate.")
        return 0.0, 0.0, 0.0, float("nan")

    y_true = valid_df["label"]
    y_pred = valid_df["tier2_label"]

    precision = precision_score(y_true, y_pred, labels=ORDINAL_ORDER, average="macro", zero_division=0)
    macro_f1  = f1_score(y_true, y_pred, labels=ORDINAL_ORDER, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=ORDINAL_ORDER, average="weighted", zero_division=0)
    ordinal_mae = mean_absolute_error_ordinal(y_true, y_pred)

    return precision, macro_f1, weighted_f1, ordinal_mae


def print_classification_report(filtered_df: pd.DataFrame) -> None:
    """Prints per-class breakdown and confusion matrix for Tier 2 predictions."""
    valid_df = filtered_df[
        filtered_df["tier2_label"].isin(ORDINAL_ORDER) &
        filtered_df["label"].isin(ORDINAL_ORDER)
    ]
    if valid_df.empty:
        return

    print("\nPer-Class Classification Report (Tier 2):")
    print(classification_report(
        valid_df["label"],
        valid_df["tier2_label"],
        labels=ORDINAL_ORDER,
        zero_division=0,
    ))

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(valid_df["label"], valid_df["tier2_label"], labels=ORDINAL_ORDER)
    header = f"{'':>10}" + "".join(f"{l:>10}" for l in ORDINAL_ORDER)
    print(header)
    for label, row in zip(ORDINAL_ORDER, cm):
        print(f"{label:>10}" + "".join(f"{v:>10}" for v in row))


# ── Report ─────────────────────────────────────────────────────────────────────

def print_final_report(
    tier1_metrics: dict,
    tier1_recall: float,
    tier2_precision: float,
    tier2_macro_f1: float,
    tier2_weighted_f1: float,
    tier2_ordinal_mae: float,
) -> None:
    print("\n" + "=" * 52)
    print("       CASCADE SYSTEM EVALUATION REPORT")
    print("=" * 52)

    print("\n[Tier 1 — Sentinel Filter]")
    print(f"  Model              : mrm8488/distilroberta-finetuned-depression")
    print(f"  Threshold          : p > 0.30")
    print(f"  Posts In / Out     : {tier1_metrics['original_count']} → {tier1_metrics['passed_count']}")
    print(f"  Reduction          : {tier1_metrics['reduction_percentage']:.1f}% of posts filtered out")
    print(f"  Latency per Post   : {tier1_metrics['latency_per_post_ms']:.2f} ms")
    print(f"  System Recall      : {tier1_recall:.1f}%  (severe cases retained)")

    print("\n[Tier 2 — Fine-tuned LLM Reasoning Engine]")
    print(f"  Base Model         : meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"  Adaptation         : QLoRA (4-bit NF4, rank=16) + WeightedRandomSampler")
    print(f"  Moderation Prec.   : {tier2_precision:.4f}  (macro)")
    print(f"  F1 Score (macro)   : {tier2_macro_f1:.4f}")
    print(f"  F1 Score (weighted): {tier2_weighted_f1:.4f}")
    print(f"  Ordinal MAE        : {tier2_ordinal_mae:.4f}  (severity-level distance)")

    print("\n" + "=" * 52 + "\n")
    