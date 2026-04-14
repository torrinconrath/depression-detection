import pandas as pd
import json
import os

def load_dsd_dataset(filepath):
    """
    Loads the Depression Severity Dataset (DSD).
    Expected columns: 'text', 'label'
    Labels: minimal, mild, moderate, severe
    """
    if not os.path.exists(filepath):
        print(f"Warning: DSD file not found at {filepath}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["text", "label"])
    
    df = pd.read_csv(filepath)
    # Standardize labels to lowercase
    df['label'] = df['label'].str.lower().str.strip()
    return df

def load_erisk_dataset(filepath):
    """
    Loads the eRisk JSON dataset.
    Extracts texts from submissions and comments.
    """
    if not os.path.exists(filepath):
        print(f"Warning: eRisk file not found at {filepath}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["text", "label"])
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    records = []
    # Simplified parsing based on provided Listing 2
    for user_data in data:
        # Submission body
        if "body" in user_data and user_data["body"].strip() != "":
            records.append({"text": user_data["body"], "label": "unknown"})
            
        # Comments
        for comment in user_data.get("comments", []):
            if "body" in comment and comment["body"].strip() != "":
                records.append({"text": comment["body"], "label": "unknown"})
                
    return pd.DataFrame(records)

def get_dummy_data():
    """Returns dummy data for testing the pipeline if real data isn't present."""
    return pd.DataFrame({
        "text": [
            "I've been feeling absolutely terrible and hopeless lately.",
            "Just had a great lunch with my friends!",
            "I'm a bit stressed about my upcoming exams but managing.",
            "I can't get out of bed, everything feels meaningless and dark."
        ],
        "label": ["severe", "minimal", "mild", "severe"]
    })
