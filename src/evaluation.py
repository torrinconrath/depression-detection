import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from src.constants import ORDINAL_ORDER, BASE_FILTER_ID


def evaluate_tier1(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> dict:
    """
    Per-class recall at Tier 1.

    Clinical priority order:
      severe   — must be retained; these users need urgent intervention
      moderate — important to retain; may need support
      mild     — useful to retain; early intervention opportunity
      minimal  — safely discardable; binary model will pass most anyway due to
                 depression-adjacent language, and Tier 2 will correctly label any
                 that pass through
    """
    recall = {}
    for label in ORDINAL_ORDER:
        total  = len(original_df[original_df["label"] == label])
        passed = len(filtered_df[filtered_df["label"] == label])
        recall[label] = (passed / total * 100) if total else 100.0

    at_risk_total  = len(original_df[original_df["label"].isin(["mild", "moderate", "severe"])])
    at_risk_passed = len(filtered_df[filtered_df["label"].isin(["mild", "moderate", "severe"])])
    recall["at_risk"] = (at_risk_passed / at_risk_total * 100) if at_risk_total else 100.0

    return recall


def evaluate_tier2(df: pd.DataFrame) -> tuple[float, float, float, float, float, dict]:
    """
    Returns macro precision, macro recall, macro F1, weighted F1, ordinal MAE,
    and a per-class dict with precision/recall/F1 for each severity level.
    Macro recall is computed directly from sklearn — no secondary derivation needed.
    """
    valid = df[df["tier2_label"].isin(ORDINAL_ORDER) & df["label"].isin(ORDINAL_ORDER)].copy()
    excluded = len(df) - len(valid)
    if excluded:
        print(f"[Eval] Excluded {excluded} rows with unparseable Tier 2 labels.")
    if valid.empty:
        return 0.0, 0.0, 0.0, 0.0, float("nan"), {}

    y_true, y_pred = valid["label"], valid["tier2_label"]
    order_map   = {l: i for i, l in enumerate(ORDINAL_ORDER)}
    ordinal_mae = float(np.mean(np.abs(y_true.map(order_map) - y_pred.map(order_map))))

    per_class = {}
    for label in ORDINAL_ORDER:
        per_class[label] = {
            "precision": float(precision_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)),
            "recall":    float(recall_score(   y_true, y_pred, labels=[label], average="macro", zero_division=0)),
            "f1":        float(f1_score(       y_true, y_pred, labels=[label], average="macro", zero_division=0)),
        }

    return (
        float(precision_score(y_true, y_pred, labels=ORDINAL_ORDER, average="macro", zero_division=0)),
        float(recall_score(   y_true, y_pred, labels=ORDINAL_ORDER, average="macro", zero_division=0)),
        float(f1_score(       y_true, y_pred, labels=ORDINAL_ORDER, average="macro", zero_division=0)),
        float(f1_score(       y_true, y_pred, labels=ORDINAL_ORDER, average="weighted", zero_division=0)),
        ordinal_mae,
        per_class,
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


def save_results_json(
    t1_metrics:     dict,
    t1_recall:      dict,
    t2_precision:   float,
    t2_macro_recall: float,
    t2_macro_f1:    float,
    t2_weighted_f1: float,
    t2_ordinal_mae: float,
    t2_per_class:   dict,
    output_path:    str = "results/eval_results.json",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = {
        "tier1": {
            "model":               BASE_FILTER_ID,
            "threshold":           t1_metrics["threshold"],
            "posts_in":            t1_metrics["original_count"],
            "posts_out":           t1_metrics["passed_count"],
            "filtered_out_pct":    round(t1_metrics["reduction_percentage"], 2),
            "throughput_per_sec":  round(t1_metrics["throughput_per_sec"], 1),
            "latency_per_post_ms": round(t1_metrics["latency_per_post_ms"], 3),
            "recall": {
                "severe":   round(t1_recall["severe"],   2),
                "moderate": round(t1_recall["moderate"], 2),
                "mild":     round(t1_recall["mild"],     2),
                "minimal":  round(t1_recall["minimal"],  2),
                "at_risk":  round(t1_recall["at_risk"],  2),
            },
        },
        "tier2": {
            "base_model":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "adaptation":  "QLoRA (4-bit NF4, rank=16)",
            "supervision": "label-only with CoT system prompt",
            "macro": {
                "precision":   round(t2_precision,     4),
                "recall":      round(t2_macro_recall,  4),
                "f1_macro":    round(t2_macro_f1,      4),
                "f1_weighted": round(t2_weighted_f1,   4),
                "ordinal_mae": round(t2_ordinal_mae, 4) if not np.isnan(t2_ordinal_mae) else None,
            },
            "per_class": {
                label: {k: round(v, 4) for k, v in metrics.items()}
                for label, metrics in t2_per_class.items()
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Results saved → '{output_path}'")


def print_final_report(
    t1_metrics:      dict,
    t1_recall:       dict,
    t2_precision:    float,
    t2_macro_recall: float,
    t2_macro_f1:     float,
    t2_weighted_f1:  float,
    t2_ordinal_mae:  float,
    t2_per_class:    dict | None = None,
) -> None:
    w = 52
    print("\n" + "=" * w)
    print("       CASCADE SYSTEM EVALUATION REPORT")
    print("=" * w)

    print(f"\n[Tier 1 — Binary Sentinel Filter]")
    print(f"  Model              : {BASE_FILTER_ID}")
    print(f"  Mode               : Binary (depressive / non-depressive)")
    print(f"  Threshold          : p > {t1_metrics['threshold']:.2f}  (recall-priority gate)")
    print(f"  Posts In / Out     : {t1_metrics['original_count']} → {t1_metrics['passed_count']}")
    print(f"  Filtered Out       : {t1_metrics['reduction_percentage']:.1f}%")
    print(f"  Throughput         : {t1_metrics['throughput_per_sec']:.0f} posts/sec")
    print(f"  Latency per Post   : {t1_metrics['latency_per_post_ms']:.2f} ms")
    print(f"\n  Per-Class Recall:")
    print(f"    severe   : {t1_recall['severe']:5.1f}%  ← must be high (intervention needed)")
    print(f"    moderate : {t1_recall['moderate']:5.1f}%  ← useful to retain")
    print(f"    mild     : {t1_recall['mild']:5.1f}%  ← helpful to retain")
    print(f"    minimal  : {t1_recall['minimal']:5.1f}%  ← safely discardable")
    print(f"  At-risk Recall     : {t1_recall['at_risk']:.1f}%  (mild + moderate + severe)")

    print(f"\n[Tier 2 — LLM Reasoning Engine]")
    print(f"  Base Model         : meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"  Adaptation         : QLoRA (4-bit NF4, rank=16), severity-biased sampler")
    print(f"  Supervision        : Label-only — model generates own CoT reasoning")

    if t2_per_class:
        print(f"\n    {'Label':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for label in ORDINAL_ORDER:
            if label in t2_per_class:
                m = t2_per_class[label]
                print(f"    {label:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print(f"\n  Aggregate Metrics:")
    print(f"    Precision (macro): {t2_precision:.4f}")
    print(f"    Recall (macro)   : {t2_macro_recall:.4f}")
    print(f"    F1 — Macro       : {t2_macro_f1:.4f}")
    print(f"    F1 — Weighted    : {t2_weighted_f1:.4f}")
    print(f"    Ordinal MAE      : {t2_ordinal_mae:.4f}  (avg severity-level distance)")

    print("\n" + "=" * w + "\n")
    