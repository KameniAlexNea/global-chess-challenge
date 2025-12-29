"""
SFT training on move sequences from Stockfish evaluations.

Training approach:
- Load full lines (sequences of best moves)
- Create multi-turn conversations where assistant plays one color
- Each turn includes FEN and legal moves for full context
- Both user and assistant respond with <uci_move>xxxx</uci_move>

This teaches the model to play full games with proper context.
"""

import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_SERVICE"] = "true"

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

model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess dataset
sft_train, sft_eval = load_sft_sequences_dataset(
    tokenizer=tokenizer,
    data_file="data/processed/move_sequences_500mb.jsonl",
    train_samples=1_000_000,
    test_size=0.01,
    max_length=1024,
    num_proc=16,
    seed=42,
)

# Load model with 4-bit quantization
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Required for 4-bit to work properly
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
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

# Training config
training_args = TrainingArguments(
    output_dir="models/chess-sft-fullsequences",
    num_train_epochs=1,
    per_device_train_batch_size=16,  # Balanced for device_map auto
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    logging_steps=50,
    save_steps=2000,
    eval_steps=1000,
    eval_strategy="steps",
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="chess-sft-fullsequences-v1",
    remove_unused_columns=False,
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_train,
    eval_dataset=sft_eval,
)

print("\n" + "=" * 80)
print("Starting SFT training on move sequences...")
print(f"Training examples: {len(sft_train):,}")
print(f"Eval examples: {len(sft_eval):,}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(
    f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
)
print("=" * 80 + "\n")

trainer.train()

# Save adapter
adapter_path = training_args.output_dir + "/adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# Merge and save full model
print("\nMerging adapter with base model...")
model = model.merge_and_unload()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print(f"\n{'='*80}")
print("SFT TRAINING COMPLETE!")
print(f"Model saved to: {training_args.output_dir}")
print(f"{'='*80}")
