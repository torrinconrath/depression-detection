"""
main.py — Two-Tier Depression Severity Detection Pipeline

Architecture:
    Tier 1 — Binary recall gate (mrm8488/distilroberta-base-finetuned-suicide-depression)
              Pretrained binary classifier, no fine-tuning required. Low threshold (p > 0.09)
              ensures 100% severe recall. Tier 1's only job is to ask "might this person
              need attention?" — severity classification is exclusively Tier 2's job.

    Tier 2 — Llama 3.1-8B-Instruct + QLoRA adapter fine-tuned on DSD (post → label).
              The system prompt instructs the model to produce clinical reasoning before
              the label. No synthetic reasoning stubs are used — the model generates its
              own CoT from its pre-trained knowledge; fine-tuning aligns the label output
              to the four-class DSD severity scale.

Run order:
    1. python -m src.tier2_finetune        (GPU required — trains the Tier 2 adapter)
    2. python main.py                      (runs the full two-tier pipeline on the test set)

    Optional: set skip_tier2: True to evaluate Tier 1 alone without the adapter.
"""

import os
from src.data_loader import load_dsd_dataset, split_dataset, print_label_distribution
from src.tier1_filter import Tier1Filter
from src.tier2_llm import Tier2ReasoningEngine
from src.evaluation import (
    evaluate_tier1, evaluate_tier2,
    print_classification_report, print_final_report, save_results_json,
)

CONFIG = {
    "data_path":        "data/dsd.csv",
    "output_path":      "data/final_results.csv",
    "results_json":     "results/eval_results.json",
    "test_size":        0.2,
    "threshold":        0.09,   # highest threshold that produced 100% severe recall for tier 1
    "adapter_path":     "models/tier2_adapter",
    "skip_tier2":       False,
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

    # Tier 1 — pretrained binary recall gate
    tier1 = Tier1Filter(threshold=CONFIG["threshold"])
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
        print_final_report(t1_metrics, t1_recall, 0.0, 0.0, 0.0, 0.0, float("nan"))
        save_results_json(t1_metrics, t1_recall, 0.0, 0.0, 0.0, 0.0, float("nan"), {},
                          output_path=CONFIG["results_json"])
        return

    if not os.path.isdir(CONFIG["adapter_path"]):
        print(f"\n[Error] No adapter found at '{CONFIG['adapter_path']}'.")
        print(f"[Error] Run: python -m src.tier2_finetune")
        return

    t2_per_class = {}
    try:
        tier2    = Tier2ReasoningEngine(adapter_path=CONFIG["adapter_path"])
        final_df = tier2.process_filtered_posts(filtered_df)

        t2_precision, t2_macro_recall, t2_macro_f1, t2_weighted_f1, t2_ordinal_mae, t2_per_class = evaluate_tier2(final_df)
        print_classification_report(final_df)

        os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
        final_df.to_csv(CONFIG["output_path"], index=False)
        print(f"\n[Output] Results saved → '{CONFIG['output_path']}'")

    except Exception as e:
        print(f"\n[Error] Tier 2 failed: {e}")
        t2_precision = t2_macro_recall = t2_macro_f1 = t2_weighted_f1 = 0.0
        t2_ordinal_mae = float("nan")

    print_final_report(t1_metrics, t1_recall, t2_precision, t2_macro_recall, t2_macro_f1,
                       t2_weighted_f1, t2_ordinal_mae, t2_per_class)
    save_results_json(t1_metrics, t1_recall, t2_precision, t2_macro_recall, t2_macro_f1,
                      t2_weighted_f1, t2_ordinal_mae, t2_per_class, output_path=CONFIG["results_json"])


if __name__ == "__main__":
    main()
