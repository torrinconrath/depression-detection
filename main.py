"""
main.py — Two-Tier Depression Severity Detection Pipeline

Run order:
    python main.py --data data/dsd.csv --skip-tier2   # sanity check, saves splits
    python -m src.tier2_finetune                       # fine-tune LLM (~1-2 hrs, GPU required)
    python main.py --data data/dsd.csv                 # full pipeline
"""

import argparse
import os

from src.data_loader import load_dsd_dataset, split_dataset, print_label_distribution
from src.tier1_filter import Tier1Filter
from src.tier2_llm import Tier2ReasoningEngine
from src.evaluation import (
    evaluate_tier1,
    evaluate_tier2,
    print_classification_report,
    print_final_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Tier Depression Severity Detection Pipeline")
    parser.add_argument("--data",         default="data/dsd.csv",
                        help="Path to the DSD CSV file")
    parser.add_argument("--threshold",    default=0.3,  type=float,
                        help="Tier 1 depressive probability threshold (default: 0.3)")
    parser.add_argument("--tier1-model",  default="mrm8488/distilroberta-base-finetuned-suicide-depression",
                        help="HuggingFace model ID for the Tier 1 sentinel filter")
    parser.add_argument("--adapter-path", default="models/tier2_adapter",
                        help="Path to the fine-tuned LoRA adapter for Tier 2")
    parser.add_argument("--output",       default="data/final_results.csv",
                        help="Path to save final results CSV")
    parser.add_argument("--test-size",    default=0.2,  type=float,
                        help="Fraction of data held out for evaluation (default: 0.2)")
    parser.add_argument("--skip-tier2",   action="store_true",
                        help="Run Tier 1 only — useful for benchmarking the filter in isolation")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 52)
    print("  Early Depression Severity Detection Pipeline")
    print("=" * 52)

    # ── 1. Load & split data ────────────────────────────────────────────────
    df = load_dsd_dataset(args.data)
    print_label_distribution(df, name="Full Dataset")

    train_df, test_df = split_dataset(df, test_size=args.test_size)
    print_label_distribution(train_df, name="Train Split")
    print_label_distribution(test_df,  name="Test Split")

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv",   index=False)
    print("\n[Data] Splits saved → data/train.csv, data/test.csv")

    # ── 2. Tier 1: Sentinel Filter ──────────────────────────────────────────
    print(f"\n[Pipeline] Tier 1 — scoring {len(test_df)} test posts...")
    tier1       = Tier1Filter(model_name=args.tier1_model, threshold=args.threshold)
    filtered_df, t1_metrics = tier1.filter_posts(test_df)

    t1_recall = evaluate_tier1(original_df=test_df, filtered_df=filtered_df)
    print(f"[Tier 1] System Recall (severe): {t1_recall:.1f}%")

    if args.skip_tier2:
        print("\n[Pipeline] --skip-tier2 set. Stopping after Tier 1.")
        print_final_report(t1_metrics, t1_recall, 0.0, 0.0, 0.0, float("nan"))
        return

    if filtered_df.empty:
        print("\n[Pipeline] Tier 1 passed 0 posts. Nothing to send to Tier 2.")
        print_final_report(t1_metrics, t1_recall, 0.0, 0.0, 0.0, float("nan"))
        return

    # ── 3. Tier 2: Fine-tuned LLM ──────────────────────────────────────────
    if not os.path.isdir(args.adapter_path):
        print(
            f"\n[Error] No adapter found at '{args.adapter_path}'.\n"
            f"  Fine-tune first:\n"
            f"  python -m src.tier2_finetune --train data/train.csv --output {args.adapter_path}"
        )
        return

    try:
        tier2    = Tier2ReasoningEngine(adapter_path=args.adapter_path)
        final_df = tier2.process_filtered_posts(filtered_df)

        t2_precision, t2_macro_f1, t2_weighted_f1, t2_ordinal_mae = evaluate_tier2(final_df)
        print_classification_report(final_df)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        final_df.to_csv(args.output, index=False)
        print(f"\n[Output] Results saved → '{args.output}'")

    except EnvironmentError as e:
        print(f"\n[Error] {e}")
        t2_precision = t2_macro_f1 = t2_weighted_f1 = 0.0
        t2_ordinal_mae = float("nan")

    except Exception as e:
        print(f"\n[Error] Tier 2 failed: {e}")
        t2_precision = t2_macro_f1 = t2_weighted_f1 = 0.0
        t2_ordinal_mae = float("nan")

    # ── 4. Final Report ─────────────────────────────────────────────────────
    print_final_report(t1_metrics, t1_recall, t2_precision, t2_macro_f1, t2_weighted_f1, t2_ordinal_mae)


if __name__ == "__main__":
    main()