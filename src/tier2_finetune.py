"""
tier2_finetune.py

QLoRA fine-tuning script for Tier 2 LLM (Llama 3.1-8B-Instruct).

This script fine-tunes the LLM on the DSD training split using:
  - QLoRA (4-bit base + LoRA adapters) to fit on a single consumer GPU
  - Chain-of-Thought formatted training examples (Reasoning: ... Label: ...)
  - WeightedRandomSampler to counteract the dataset's severe class imbalance
    (minimal ~72.8%, severe ~7.9%)

Run BEFORE main.py:
    python -m src.tier2_finetune --train data/train.csv --output models/tier2_adapter

Requirements:
    pip install transformers peft bitsandbytes trl accelerate datasets
    huggingface-cli login   # needed for gated Llama 3.1 weights
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

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ORDINAL_ORDER = ["minimal", "mild", "moderate", "severe"]

# ── CoT prompt template ────────────────────────────────────────────────────────
# Each training example is formatted as a full instruction + CoT response.
# The model learns to produce reasoning BEFORE the label, implementing
# Chain-of-Thought as described in the project proposal.
SYSTEM_PROMPT = """You are an expert clinical psychologist specialising in depression assessment.
Analyse the social media post below and classify the author's depression severity.

First provide step-by-step clinical reasoning citing specific evidence from the post.
Then state the final label.

Severity scale:
- Minimal: Little to no depressive indicators. Normal daily functioning.
- Mild: Occasional low mood or stress. Some negative affect but generally coping.
- Moderate: Persistent low mood, loss of interest, some functional impairment.
- Severe: Intense hopelessness, anhedonia, significant impairment, possible suicidal ideation.

Response format (follow exactly):
Reasoning: <2-4 sentences of clinical reasoning>
Label: <Minimal|Mild|Moderate|Severe>"""


def format_training_example(row: dict) -> dict:
    """
    Converts a DSD row into a full instruction-tuning prompt.
    The 'response' is a CoT stub — reasoning is synthesised from the label
    since the DSD does not include gold reasoning chains.

    In a production setting, a senior clinician would annotate reasoning chains.
    Here we use a structured label-derived reasoning template as a training signal,
    which still teaches the model the expected output format and label vocabulary.
    """
    label = row["label"].capitalize()
    text = row["text"]

    # Label-conditioned reasoning stubs that teach format without hallucinating
    # clinician knowledge the model doesn't have from the text alone.
    reasoning_stubs = {
        "Minimal": "The post does not contain prominent depressive language. "
                   "The author appears to be functioning normally with no marked signs of emotional distress. "
                   "Affect and tone are broadly neutral or positive.",
        "Mild": "The post contains some indicators of low mood or stress, but the author appears to be coping. "
                "Negative affect is present but not pervasive. "
                "There is no evidence of significant functional impairment.",
        "Moderate": "The post shows persistent low mood and reduced interest or energy. "
                    "There are signs of some functional impairment in daily life. "
                    "The language suggests ongoing distress beyond typical stress responses.",
        "Severe": "The post contains strong indicators of hopelessness, anhedonia, or significant functional breakdown. "
                  "The author's language suggests intense and pervasive distress. "
                  "There may be implicit or explicit indicators of risk.",
    }

    reasoning = reasoning_stubs.get(label, "Insufficient information to determine severity.")

    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\nPost: \"{text}\"<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\nReasoning: {reasoning}\nLabel: {label}<|eot_id|>"

    return {"text": prompt}


def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler so each class is sampled equally per epoch,
    directly counteracting the minimal ~72.8% dominance during training.
    """
    label_counts = train_df["label"].value_counts().to_dict()
    total = len(train_df)
    # Weight for each sample = inverse of its class frequency
    sample_weights = train_df["label"].map(
        lambda lbl: total / (len(ORDINAL_ORDER) * label_counts.get(lbl, 1))
    ).tolist()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total,
        replacement=True,
    )


def finetune(train_csv: str, output_dir: str, num_epochs: int = 3) -> None:
    """
    Main fine-tuning routine.

    Args:
        train_csv:   Path to the training split CSV (from split_dataset).
        output_dir:  Directory to save the LoRA adapter weights.
        num_epochs:  Number of training epochs (default 3; increase for better convergence).
    """
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "QLoRA fine-tuning requires a CUDA GPU. "
            "Run on a machine with a GPU (>=16 GB VRAM recommended for Llama 3.1-8B)."
        )

    print(f"[Finetune] Loading training data from '{train_csv}'...")
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df["label"].isin(ORDINAL_ORDER)].reset_index(drop=True)
    print(f"[Finetune] {len(train_df)} training examples.")

    # Format all examples into CoT instruction strings
    formatted = train_df.apply(format_training_example, axis=1).tolist()
    hf_dataset = Dataset.from_list(formatted)

    # ── Load model in 4-bit ──────────────────────────────────────────────────
    print(f"[Finetune] Loading base model: {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for SFT causal LM training

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # ── Attach LoRA adapters ─────────────────────────────────────────────────
    # Targeting attention + MLP projection layers for best coverage/cost tradeoff
    lora_config = LoraConfig(
        r=16,                        # Rank — higher = more parameters, better fit
        lora_alpha=32,               # Scaling factor (typically 2x rank)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # Effective batch size = 16
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,              # Keep only the best checkpoint
        optim="paged_adamw_8bit",        # Memory-efficient optimizer for QLoRA
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",                # Disable wandb/tensorboard unless configured
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    # Note: SFTTrainer handles the causal LM loss masking automatically.
    # The WeightedRandomSampler is injected via a custom subclass below.
    trainer = _WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=768,
        train_df=train_df,             # Passed through for sampler construction
    )

    print(f"[Finetune] Starting QLoRA fine-tuning for {num_epochs} epochs...")
    trainer.train()

    print(f"[Finetune] Saving adapter to '{output_dir}'...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Finetune] Done. Adapter saved.")


class _WeightedSFTTrainer(SFTTrainer):
    """
    SFTTrainer subclass that injects a WeightedRandomSampler so minority
    severity classes (mild, moderate, severe) are upsampled during training.
    """

    def __init__(self, *args, train_df: pd.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_df = train_df

    def _get_train_sampler(self):
        return build_weighted_sampler(self._train_df)


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Tier 2 LLM on DSD training split")
    parser.add_argument("--train", type=str, default="data/train.csv", help="Training CSV path")
    parser.add_argument("--output", type=str, default="models/tier2_adapter", help="Adapter output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    finetune(train_csv=args.train, output_dir=args.output, num_epochs=args.epochs)