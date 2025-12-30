"""
FASTER SFT training - no gradient checkpointing, larger batches.
"""

import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.sft_data import load_sft_sequences_dataset

model_name = "unsloth/gemma-3-270m-it-unsloth-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
sft_train, sft_eval = load_sft_sequences_dataset(
    tokenizer=tokenizer,
    data_file="data/processed/move_sequences_500mb.jsonl",
    train_samples=500_000,
    test_size=0.01,
    max_length=2048,  # Reduced from 1024 for speed
    num_proc=16,
    seed=42,
)

# Load model
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map={"": 0},
)

model = prepare_model_for_kbit_training(model)

# LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training config - NO GRADIENT CHECKPOINTING
training_args = TrainingArguments(
    output_dir="models/chess-sft-conversation",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Larger batch
    gradient_accumulation_steps=16,  # No accumulation
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    logging_steps=50,
    save_steps=1000,
    eval_steps=1000,
    eval_strategy="steps",
    bf16=True,
    gradient_checkpointing=False,  # DISABLED - much faster
    report_to="wandb",
    run_name="chess-sft-conversation",
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_train,
    eval_dataset=sft_eval,
)

print("\n" + "=" * 80)
print("Starting FAST SFT training (no gradient checkpointing)...")
print(f"Training examples: {len(sft_train):,}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print("=" * 80 + "\n")

trainer.train()

# Save
adapter_path = training_args.output_dir + "/adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

model = model.merge_and_unload()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print("Training complete!")
