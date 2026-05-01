"""
tier2_finetune.py — QLoRA fine-tuning for the Tier 2 LLM.

Fine-tunes Llama 3.1-8B-Instruct on the DSD training split using:
  - QLoRA (4-bit NF4 base + LoRA adapters) to fit on a single consumer GPU
  - Label-only supervision: the assistant turn contains only "Label: <Severity>".
    The system prompt instructs the model to produce Reasoning before the Label at
    both train and inference time, so the model generates its own clinical CoT from
    its pre-trained knowledge. We do not supervise the reasoning content because
    synthetic stubs have no connection to the actual post text — training on them
    causes a train/inference mismatch where the model learns stub patterns rather
    than reasoning from post content.
  - Severity-biased sampler: severe 8x, moderate 3x, mild 2.5x boost over natural
    frequency to counter the 72.8% minimal prior in the DSD dataset.

Usage:
    python -m src.tier2_finetune
"""

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from src.constants import BASE_MODEL_ID, ORDINAL_ORDER, SYSTEM_PROMPT

CONFIG = {
    "train_csv":   "data/train.csv",
    "output_dir":  "models/tier2_adapter",
    "epochs":      3, # started at 4, but noticed overfitting with ideal checkpoint at epoch 3
    "max_seq_len": 384, # lowest token length to capture the full sentence within the dataset (check_token_length.py)
}

# Per-class sampling multipliers to counter the 72.8% minimal prior.
# Effective batch composition becomes roughly:
#   minimal ~35%, mild ~18%, moderate ~20%, severe ~27%
SAMPLE_WEIGHTS = {"minimal": 1.0, "mild": 2.5, "moderate": 3.0, "severe": 8.0}


def format_example(row: pd.Series) -> dict:
    """
    Format a training example as a ChatML-style prompt with label-only supervision.

    The assistant turn contains only "Label: <Severity>" — no synthetic reasoning.
    The system prompt's Reasoning:/Label: format means the model still learns to
    generate CoT at inference; fine-tuning solely aligns the final label prediction
    to the DSD severity scale using the model's own pre-trained clinical knowledge.
    """
    label = row["label"].capitalize()
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + SYSTEM_PROMPT
        + "<|eot_id|>"
        + "<|start_header_id|>user<|end_header_id|>\n"
        + f"Post: \"{row['text']}\""
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>\n"
        + f"Label: {label}"
        + "<|eot_id|>"
    )
    return {"text": prompt}


def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    weights = train_df["label"].map(lambda lbl: SAMPLE_WEIGHTS.get(lbl, 1.0)).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=len(train_df), replacement=True)


class _WeightedTrainer(Trainer):
    """Trainer subclass that injects the severity-biased sampler."""
    def __init__(self, *args, train_df: pd.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_df = train_df

    def _get_train_sampler(self):
        return build_weighted_sampler(self._train_df)


def finetune() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("QLoRA fine-tuning requires a CUDA GPU.")

    train_df = pd.read_csv(CONFIG["train_csv"])
    train_df = train_df[train_df["label"].isin(ORDINAL_ORDER)].reset_index(drop=True)
    print(f"[Finetune] {len(train_df)} training examples loaded.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    hf_dataset = Dataset.from_list(train_df.apply(format_example, axis=1).tolist()).map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=CONFIG["max_seq_len"]),
        batched=True, remove_columns=["text"],
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print(f"[Finetune] Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = get_peft_model(
        prepare_model_for_kbit_training(base_model),
        LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    _WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=CONFIG["output_dir"],
            num_train_epochs=CONFIG["epochs"],
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            #save_strategy="epoch", # turned off due to computer freezing during savng checkpoints (my system is dying)
            save_strategy="no",
            save_total_limit=1,
            optim="paged_adamw_8bit",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
        ),
        train_dataset=hf_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_df=train_df,
    ).train()

    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"[Finetune] Adapter saved to '{CONFIG['output_dir']}'.")


if __name__ == "__main__":
    print("=" * 52 + "\n  Tier 2 LLM Finetuning\n" + "=" * 52)
    finetune()
