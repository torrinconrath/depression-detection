"""
tier2_finetune.py — QLoRA fine-tuning for the Tier 2 LLM.

Fine-tunes Llama 3.1-8B-Instruct on the DSD training split using:
  - QLoRA (4-bit NF4 base + LoRA adapters) to fit on a single consumer GPU
  - Chain-of-Thought formatted examples (Reasoning: ... Label: ...)
  - WeightedRandomSampler to counter class imbalance
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
from src.constants import BASE_MODEL_ID, ORDINAL_ORDER, SYSTEM_PROMPT, COT_STUBS

CONFIG = {
    "train_csv":   "data/train.csv",
    "output_dir":  "models/tier2_adapter",
    "epochs":      3,
    "max_seq_len": 384, # used check_token_length to find max ideal
}


def format_example(row: pd.Series) -> dict:
    label     = row["label"].capitalize()
    reasoning = COT_STUBS.get(label, "Insufficient information to determine severity.")
    prompt  = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\nPost: \"{row['text']}\"<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\nReasoning: {reasoning}\nLabel: {label}<|eot_id|>"
    return {"text": prompt}


def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    counts  = train_df["label"].value_counts().to_dict()
    total   = len(train_df)
    weights = train_df["label"].map(lambda lbl: total / (len(ORDINAL_ORDER) * counts.get(lbl, 1))).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


class _WeightedTrainer(Trainer):
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
            save_strategy="epoch",
            save_total_limit=1,
            optim="paged_adamw_8bit",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
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
