"""
main.py — Two-Tier Depression Severity Detection Pipeline

Workflow:
  1. Load & split DSD dataset (stratified 80/20)
  2. [Optional] Fine-tune Tier 2 LLM via QLoRA  →  python -m src.tier2_finetune
  3. Run Tier 1 sentinel filter on the test split
  4. Run Tier 2 fine-tuned LLM on Tier 1 output
  5. Evaluate and report

Usage:
  # First time: fine-tune the LLM (requires GPU, ~1-2 hrs)
  python -m src.tier2_finetune --train data/train.csv --output models/tier2_adapter

  # Then run the full pipeline
  python main.py --data data/dsd.csv

  # Skip Tier 2 to benchmark Tier 1 only
  python main.py --data data/dsd.csv --skip-tier2

  # Smoke-test with dummy data (no real dataset needed)
  python main.py --dummy
"""

import argparse
import os

from src.data_loader import (
    get_dummy_data,
    load_dsd_dataset,
    print_label_distribution,
    split_dataset,
)
from src.evaluation import (
    evaluate_tier1,
    evaluate_tier2,
    print_classification_report,
    print_final_report,
)
from src.tier1_filter import Tier1Filter
from src.tier2_llm import Tier2ReasoningEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Tier Depression Severity Detection Pipeline")
    parser.add_argument("--data", type=str, default="data/dsd.csv",
                        help="Path to the DSD CSV file")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Tier 1 depressive probability threshold (default: 0.3)")
    parser.add_argument("--tier1-model", type=str,
                        default="mrm8488/distilroberta-base-finetuned-suicide-depression",
                        help="HuggingFace model ID for the Tier 1 sentinel filter")
    parser.add_argument("--adapter-path", type=str, default="models/tier2_adapter",
                        help="Path to the fine-tuned LoRA adapter for Tier 2")
    parser.add_argument("--output", type=str, default="data/final_results.csv",
                        help="Path to save final results CSV")
    parser.add_argument("--skip-tier2", action="store_true",
                        help="Run Tier 1 only (useful for benchmarking the filter)")
    parser.add_argument("--dummy", action="store_true",
                        help="Use built-in dummy dataset for a quick smoke-test")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for evaluation (default: 0.2)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 52)
    print("  Early Depression Severity Detection Pipeline")
    print("=" * 52)

    # ── 1. Load Data ────────────────────────────────────────────────────────
    if args.dummy:
        print("\n[Data] Using built-in dummy dataset (--dummy flag set).")
        df = get_dummy_data()
        test_df = df       # Use all dummy data as test — no split needed
        train_df = df
    else:
        df = load_dsd_dataset(args.data)
        if df.empty:
            print("[Data] Dataset not found or empty — falling back to dummy data.")
            df = get_dummy_data()

        print_label_distribution(df, dataset_name="Full Dataset")

        # Stratified split — preserves the ~72.8%/8% minimal/severe ratio
        train_df, test_df = split_dataset(df, test_size=args.test_size)
        print_label_distribution(train_df, dataset_name="Train Split")
        print_label_distribution(test_df, dataset_name="Test Split")

        # Save splits so tier2_finetune.py can consume train.csv independently
        os.makedirs("data", exist_ok=True)
        train_df.to_csv("data/train.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        print("\n[Data] Splits saved to data/train.csv and data/test.csv")
        print("[Data] Run 'python -m src.tier2_finetune --train data/train.csv' to fine-tune Tier 2.")

    # ── 2. Tier 1: Sentinel Filter ──────────────────────────────────────────
    print(f"\n[Pipeline] Starting Tier 1 filter on {len(test_df)} test posts...")
    tier1 = Tier1Filter(model_name=args.tier1_model, threshold=args.threshold)
    filtered_df, t1_metrics = tier1.filter_posts(test_df)

    tier1_recall = evaluate_tier1(original_df=test_df, filtered_df=filtered_df)
    print(f"[Tier 1] System Recall (severe): {tier1_recall:.1f}%")

    if args.skip_tier2:
        print("\n[Pipeline] --skip-tier2 set. Stopping after Tier 1.")
        print_final_report(t1_metrics, tier1_recall, 0.0, 0.0, 0.0, float("nan"))
        return

    if filtered_df.empty:
        print("\n[Pipeline] Tier 1 passed 0 posts. Nothing to send to Tier 2.")
        print_final_report(t1_metrics, tier1_recall, 0.0, 0.0, 0.0, float("nan"))
        return

    # ── 3. Tier 2: Fine-tuned LLM Reasoning Engine ─────────────────────────
    print(f"\n[Pipeline] Starting Tier 2 reasoning engine...")

    if not os.path.isdir(args.adapter_path):
        print(
            f"[Error] Adapter not found at '{args.adapter_path}'.\n"
            f"  Run fine-tuning first:\n"
            f"  python -m src.tier2_finetune --train data/train.csv --output {args.adapter_path}"
        )
        return

    try:
        tier2 = Tier2ReasoningEngine(adapter_path=args.adapter_path)
        final_df = tier2.process_filtered_posts(filtered_df)

        # ── 4. Evaluate ──────────────────────────────────────────────────────
        t2_precision, t2_macro_f1, t2_weighted_f1, t2_ordinal_mae = evaluate_tier2(final_df)
        print_classification_report(final_df)

        # ── 5. Save ──────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        final_df.to_csv(args.output, index=False)
        print(f"\n[Output] Results saved to '{args.output}'")

    except EnvironmentError as e:
        print(f"\n[Error] {e}")
        t2_precision = t2_macro_f1 = t2_weighted_f1 = 0.0
        t2_ordinal_mae = float("nan")

    except Exception as e:
        print(f"\n[Error] Tier 2 failed: {e}")
        t2_precision = t2_macro_f1 = t2_weighted_f1 = 0.0
        t2_ordinal_mae = float("nan")

    # ── 6. Final Report ─────────────────────────────────────────────────────
    print_final_report(
        t1_metrics, tier1_recall,
        t2_precision, t2_macro_f1, t2_weighted_f1, t2_ordinal_mae,
    )


if __name__ == "__main__":
    main()