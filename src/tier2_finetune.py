"""
tier2_finetune.py  —  QLoRA fine-tuning for the Tier 2 LLM.

Fine-tunes Llama 3.1-8B-Instruct on the DSD training split using:
  - QLoRA (4-bit NF4 base + LoRA adapters) to fit on a single consumer GPU
  - Chain-of-Thought formatted examples  (Reasoning: ...  Label: ...)
  - WeightedRandomSampler to counter class imbalance (minimal ~72.8%, severe ~7.9%)

Run once before main.py:
    python -m src.tier2_finetune --train data/train.csv --output models/tier2_adapter

Requirements:
    pip install transformers peft bitsandbytes trl accelerate datasets
    huggingface-cli login        # Llama 3.1 is a gated model
"""

import argparse
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ── Shared constants (must match tier2_llm.py) ────────────────────────────────
BASE_MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ORDINAL_ORDER  = ["minimal", "mild", "moderate", "severe"]
SYSTEM_PROMPT  = (
    "You are an expert clinical psychologist specialising in depression assessment.\n"
    "Analyse the social media post below and classify the author's depression severity.\n\n"
    "First provide step-by-step clinical reasoning citing specific evidence from the post.\n"
    "Then state the final label.\n\n"
    "Severity scale:\n"
    "- Minimal:  Little to no depressive indicators. Normal daily functioning.\n"
    "- Mild:     Occasional low mood or stress. Some negative affect but generally coping.\n"
    "- Moderate: Persistent low mood, loss of interest, some functional impairment.\n"
    "- Severe:   Intense hopelessness, anhedonia, significant impairment, possible suicidal ideation.\n\n"
    "Response format (follow exactly):\n"
    "Reasoning: <2-4 sentences of clinical reasoning>\n"
    "Label: <Minimal|Mild|Moderate|Severe>"
)

# Label-conditioned CoT stubs.
# The DSD has no gold reasoning chains, so we use structured stubs that teach
# the model the expected output format and label vocabulary. In a production
# setting these would be replaced with clinician-annotated reasoning.
COT_STUBS = {
    "Minimal":  ("The post contains no prominent depressive language. "
                 "The author appears to be functioning normally with no marked signs of distress. "
                 "Affect and tone are broadly neutral or positive."),
    "Mild":     ("The post contains some indicators of low mood or stress, but the author appears to be coping. "
                 "Negative affect is present but not pervasive. "
                 "There is no evidence of significant functional impairment."),
    "Moderate": ("The post shows persistent low mood and reduced interest or energy. "
                 "There are signs of some functional impairment in daily life. "
                 "The language suggests ongoing distress beyond typical stress responses."),
    "Severe":   ("The post contains strong indicators of hopelessness, anhedonia, or significant functional breakdown. "
                 "The author's language suggests intense and pervasive distress. "
                 "There may be implicit or explicit indicators of risk."),
}


# ── Training data formatting ──────────────────────────────────────────────────

def format_example(row: pd.Series) -> dict:
    """Wraps a DSD row into a full Llama 3 instruction-tuning prompt with CoT response."""
    label     = row["label"].capitalize()
    reasoning = COT_STUBS.get(label, "Insufficient information to determine severity.")

    prompt  = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\nPost: \"{row['text'][:600]}\"<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\nReasoning: {reasoning}\nLabel: {label}<|eot_id|>"
    return {"text": prompt}


# ── Weighted sampler ──────────────────────────────────────────────────────────

def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Up-samples minority classes so each severity level appears equally per epoch,
    directly countering the ~72.8% minimal dominance.
    """
    counts  = train_df["label"].value_counts().to_dict()
    total   = len(train_df)
    weights = train_df["label"].map(
        lambda lbl: total / (len(ORDINAL_ORDER) * counts.get(lbl, 1))
    ).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


class _WeightedSFTTrainer(SFTTrainer):
    """SFTTrainer with WeightedRandomSampler injected for class-balanced batches."""
    def __init__(self, *args, train_df: pd.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_df = train_df

    def _get_train_sampler(self):
        return build_weighted_sampler(self._train_df)


# ── Main fine-tuning routine ──────────────────────────────────────────────────

def finetune(train_csv: str, output_dir: str, num_epochs: int = 3) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "QLoRA fine-tuning requires a CUDA GPU (>=16 GB VRAM recommended). "
            "Run on a GPU machine or cloud instance."
        )

    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df["label"].isin(ORDINAL_ORDER)].reset_index(drop=True)
    print(f"[Finetune] {len(train_df)} training examples loaded from '{train_csv}'.")

    hf_dataset = Dataset.from_list(train_df.apply(format_example, axis=1).tolist())

    # 4-bit quantised base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[Finetune] Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA adapters — attention + MLP projections
    model = get_peft_model(base_model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = _WeightedSFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,   # effective batch = 16
            learning_rate=2e-4,
            fp16=True,
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=1,
            optim="paged_adamw_8bit",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
        ),
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=768,
        train_df=train_df,
    )

    print(f"[Finetune] Starting QLoRA fine-tuning ({num_epochs} epochs)...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Finetune] Adapter saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   default="data/train.csv",        help="Training CSV path")
    parser.add_argument("--output",  default="models/tier2_adapter",  help="Adapter output directory")
    parser.add_argument("--epochs",  default=3, type=int,             help="Training epochs")
    args = parser.parse_args()

    finetune(train_csv=args.train, output_dir=args.output, num_epochs=args.epochs)
    