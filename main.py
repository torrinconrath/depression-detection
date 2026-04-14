import pandas as pd
from src.data_loader import load_dsd_dataset, get_dummy_data
from src.tier1_filter import Tier1Filter
from src.tier2_llm import Tier2ReasoningEngine
from src.evaluation import evaluate_tier1, evaluate_tier2, print_final_report

def main():
    print("Starting Early Detection Pipeline...")
    
    # 1. Load Data
    # Attempt to load DSD data. If missing, use dummy data.
    dsd_path = "data/dsd.csv"
    df = load_dsd_dataset(dsd_path)
    if len(df) == 0:
        print("Using dummy dataset for demonstration...")
        df = get_dummy_data()
        
    print(f"Loaded {len(df)} posts for processing.")

    # 2. Tier 1: Sentinel Filter
    # Threshold p > 0.3 as per project proposal
    tier1 = Tier1Filter(model_name="distilbert-base-uncased", threshold=0.3)
    filtered_df, t1_metrics = tier1.filter_posts(df)
    
    t1_recall = evaluate_tier1(original_df=df, filtered_df=filtered_df)

    print(f"Tier 1 complete. Kept {t1_metrics['passed_count']} out of {t1_metrics['original_count']} posts.")

    # 3. Tier 2: LLM Reasoning Engine
    # Note: Requires a GPU and HF login to run successfully.
    # If running on CPU/No VRAM, you might want to mock the analyze_post method.
    try:
        tier2 = Tier2ReasoningEngine()
        final_df = tier2.process_filtered_posts(filtered_df)
        
        # 4. Evaluate Tier 2
        t2_precision, t2_f1 = evaluate_tier2(final_df)
        
        # Save results to analyze Chain-of-Thought evidence
        final_df.to_csv("data/final_results.csv", index=False)
        print("Results saved to data/final_results.csv")
        
    except Exception as e:
        print(f"\n[Error] Could not run Tier 2 LLM. Make sure you have a GPU and ran 'huggingface-cli login'.")
        print(f"Error details: {e}")
        t2_precision, t2_f1 = 0.0, 0.0

    # 5. Output Report
    print_final_report(t1_metrics, t1_recall, t2_precision, t2_f1)

if __name__ == "__main__":
    main()
    