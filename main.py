"""
main.py — Two-Tier Depression Severity Detection Pipeline

Architecture:
    Tier 1 — Binary recall gate (DistilBERT fine-tuned on DSD training split)
              Fine-tuned binary classifier: negative = minimal, positive = mild/moderate/severe.
              Trained on the same domain and distribution as the test set, avoiding the
              domain-shift risk of generic pretrained suicide/depression models.
              Low threshold (p > 0.3) prioritises recall — Tier 1's only job is to ask
              "might this person need attention?"; severity classification is exclusively
              Tier 2's responsibility.

    Tier 2 — Llama 3.1-8B-Instruct + QLoRA adapter fine-tuned on DSD (post → label).
              The system prompt instructs the model to produce clinical reasoning before
              the label. No synthetic reasoning stubs are used — the model generates its
              own CoT from its pre-trained knowledge; fine-tuning aligns the label output
              to the four-class DSD severity scale.

Run order:
    1. python -m src.tier1_finetune        (trains the Tier 1 binary filter — CPU-capable)
    2. python -m src.tier2_finetune        (GPU required — trains the Tier 2 adapter)
    3. python main.py                      (runs the full two-tier pipeline on the test set)

    Optional: set skip_tier2: True to evaluate Tier 1 alone without the adapter.
"""

import os
from src.data_loader import load_dsd_dataset, split_dataset, print_label_distribution
from src.tier1_filter import Tier1Filter
from src.tier2_llm import Tier2ReasoningEngine
from src.evaluation import evaluate_tier1, evaluate_tier2, print_classification_report, print_final_report

CONFIG = {
    "data_path":        "data/dsd.csv",
    "output_path":      "data/final_results.csv",
    "test_size":        0.2,
    "threshold":        0.1,
    "tier1_model_dir":  "models/tier1_filter",
    "adapter_path":     "models/tier2_adapter",
    "skip_tier2":       True,
}


def main():
    print("=" * 52 + "\n  Early Depression Severity Detection Pipeline\n" + "=" * 52)

    df = load_dsd_dataset(CONFIG["data_path"])
    print_label_distribution(df, name="Full Dataset")

    train_df, test_df = split_dataset(df, test_size=CONFIG["test_size"])
    print_label_distribution(train_df, name="Train Split")
    print_label_distribution(test_df,  name="Test Split")

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv",   index=False)
    print("\n[Data] Splits saved → data/train.csv, data/test.csv")

    # Tier 1 — fine-tuned binary recall gate
    if not os.path.isdir(CONFIG["tier1_model_dir"]):
        print(f"\n[Error] No Tier 1 model found at '{CONFIG['tier1_model_dir']}'.")
        print(f"[Error] Run: python -m src.tier1_finetune")
        return

    tier1 = Tier1Filter(threshold=CONFIG["threshold"], model_dir=CONFIG["tier1_model_dir"])
    filtered_df, t1_metrics = tier1.filter_posts(test_df)
    t1_recall = evaluate_tier1(original_df=test_df, filtered_df=filtered_df)
    print(
        f"[Tier 1] Severe Recall: {t1_recall['severe']:.1f}% | "
        f"At-risk Recall: {t1_recall['at_risk']:.1f}% | "
        f"Throughput: {t1_metrics['throughput_per_sec']:.0f} posts/sec"
    )

    if CONFIG["skip_tier2"] or filtered_df.empty:
        reason = "skip_tier2 is True" if CONFIG["skip_tier2"] else "Tier 1 passed 0 posts"
        print(f"\n[Pipeline] {reason}. Stopping after Tier 1.")
        print_final_report(t1_metrics, t1_recall, 0.0, 0.0, 0.0, float("nan"))
        return

    if not os.path.isdir(CONFIG["adapter_path"]):
        print(f"\n[Error] No adapter found at '{CONFIG['adapter_path']}'.")
        print(f"[Error] Run: python -m src.tier2_finetune")
        return

    try:
        tier2    = Tier2ReasoningEngine(adapter_path=CONFIG["adapter_path"])
        final_df = tier2.process_filtered_posts(filtered_df)

        t2_precision, t2_macro_f1, t2_weighted_f1, t2_ordinal_mae = evaluate_tier2(final_df)
        print_classification_report(final_df)

        os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
        final_df.to_csv(CONFIG["output_path"], index=False)
        print(f"\n[Output] Results saved → '{CONFIG['output_path']}'")

    except Exception as e:
        print(f"\n[Error] Tier 2 failed: {e}")
        t2_precision = t2_macro_f1 = t2_weighted_f1 = 0.0
        t2_ordinal_mae = float("nan")

    print_final_report(t1_metrics, t1_recall, t2_precision, t2_macro_f1, t2_weighted_f1, t2_ordinal_mae)


if __name__ == "__main__":
    main()
    