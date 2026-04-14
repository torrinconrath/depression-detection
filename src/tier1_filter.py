import torch
import time
from transformers import pipeline

class Tier1Filter:
    def __init__(self, model_name="distilbert-base-uncased", threshold=0.3):
        """
        Initializes the lightweight Tier 1 filter. 
        Note: In practice, this model should be fine-tuned on binary classification 
        (Depressive vs Non-Depressive) before use.
        """
        self.threshold = threshold
        self.device = 0 if torch.cuda.is_available() else -1
        # We use a text-classification pipeline. 
        self.classifier = pipeline("text-classification", model=model_name, device=self.device)
        print(f"Tier 1 Filter loaded: {model_name} on device {self.device}")

    def filter_posts(self, df):
        """
        Processes a dataframe of texts.
        Returns:
            - filtered_df: DataFrame of posts passing the threshold.
            - metrics: Latency and reduction statistics.
        """
        start_time = time.time()
        
        texts = df['text'].tolist()
        # In a real fine-tuned model, LABEL_1 would be the "Depressive" class
        # For this base skeleton, we'll extract the raw score.
        results = self.classifier(texts, truncation=True, max_length=512)
        
        suspect_indices = []
        scores = []
        
        for i, res in enumerate(results):
            # Assuming 'score' is the probability of the positive class for a fine-tuned model
            prob = res['score'] 
            scores.append(prob)
            if prob > self.threshold:
                suspect_indices.append(i)
                
        end_time = time.time()
        
        # Calculate Latency per Post
        total_time_ms = (end_time - start_time) * 1000
        latency_per_post = total_time_ms / len(texts) if texts else 0
        
        filtered_df = df.iloc[suspect_indices].copy()
        filtered_df['tier1_score'] = [scores[i] for i in suspect_indices]
        
        reduction_percentage = ((len(df) - len(filtered_df)) / len(df)) * 100 if len(df) > 0 else 0
        
        metrics = {
            "latency_per_post_ms": latency_per_post,
            "reduction_percentage": reduction_percentage,
            "original_count": len(df),
            "passed_count": len(filtered_df)
        }
        
        return filtered_df, metrics
    