from sklearn.metrics import precision_score, f1_score

def evaluate_tier1(original_df, filtered_df):
    """
    Calculates Tier 1 System Recall: % of 'severe' labels successfully passed.
    """
    total_severe = len(original_df[original_df['label'] == 'severe'])
    passed_severe = len(filtered_df[filtered_df['label'] == 'severe'])
    
    system_recall = (passed_severe / total_severe) * 100 if total_severe > 0 else 100.0
    return system_recall

def evaluate_tier2(filtered_df):
    """
    Calculates Moderation Precision and F1 Score for Tier 2.
    Labels: minimal, mild, moderate, severe
    """
    # Filter out entries that didn't get a proper label due to parsing errors
    valid_df = filtered_df[filtered_df['tier2_label'].isin(['minimal', 'mild', 'moderate', 'severe'])]
    
    y_true = valid_df['label']
    y_pred = valid_df['tier2_label']
    
    if len(y_true) == 0:
        return 0.0, 0.0
        
    # Moderation Precision (Macro)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    # F1 Score (Macro) - Serves as the general metric
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return precision, f1

def print_final_report(tier1_metrics, tier1_recall, tier2_precision, tier2_f1):
    print("\n" + "="*40)
    print("      CASCADE SYSTEM EVALUATION REPORT")
    print("="*40)
    print("Tier 1 (Filter System):")
    print(f" - Latency per Post:    {tier1_metrics['latency_per_post_ms']:.2f} ms")
    print(f" - Reduction in Posts:  {tier1_metrics['reduction_percentage']:.2f} %")
    print(f" - System Recall:       {tier1_recall:.2f} % (Severe cases retained)")
    print("\nTier 2 (Reasoning Engine):")
    print(f" - Moderation Precision:{tier2_precision:.4f}")
    print(f" - F1 Score:            {tier2_f1:.4f}")
    print("="*40 + "\n")
    