"""
SFT training on move sequences from Stockfish evaluations.

Training approach:
- Load full lines (sequences of best moves)
- On-the-fly: randomly pick split point in line
- Create example: position after N moves → predict move N+1
- Output format: <uci_move>e2e4</uci_move>

This teaches move prediction without forced rationales.
"""

import os
os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_SERVICE"] = "true"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import chess
import random
from typing import Dict


model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("json", data_files="data/processed/move_sequences_500mb.jsonl")
dataset = dataset['train'].train_test_split(test_size=0.01, seed=42)

print(f"Loaded {len(dataset['train'])} train | {len(dataset['test'])} test lines")


def create_training_example(line_data: Dict) -> Dict:
    """
    Randomly pick a split point in the line and create training example.
    
    This is called during training, so same line generates different examples
    across epochs for better diversity.
    """
    fen = line_data['fen']
    moves = line_data['line'].split()
    
    # Randomly pick where to split (we want to predict move at split_idx)
    # Can predict any move from index 0 to len(moves)-1
    split_idx = random.randint(0, len(moves) - 1)
    
    # Apply moves up to split_idx to get current position
    board = chess.Board(fen)
    for i in range(split_idx):
        move = chess.Move.from_uci(moves[i])
        board.push(move)
    
    # Get next move to predict
    next_move = moves[split_idx]
    
    # Get legal moves
    legal_moves = [m.uci() for m in board.legal_moves]
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    
    # Create prompt
    prompt_text = f"""Analyze this chess position and find the BEST move.

Position (FEN): {board.fen()}
Side to move: {side_to_move}
Legal moves: {' '.join(legal_moves)}

Provide the best move in <uci_move> tags."""
    
    # Create response
    response = f"<uci_move>{next_move}</uci_move>"
    
    return {
        "prompt": prompt_text,
        "response": response
    }


def format_for_sft(examples):
    """Format for supervised learning."""
    texts = []
    
    for i in range(len(examples['fen'])):
        line_data = {
            'fen': examples['fen'][i],
            'line': examples['line'][i],
            'depth': examples['depth'][i]
        }
        
        # Create training example with random split
        example = create_training_example(line_data)
        
        # Format as chat
        messages = [
            {"role": "user", "content": example['prompt']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Full text
        full_text = prompt + example['response'] + tokenizer.eos_token
        texts.append(full_text)
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    # Set labels
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs


# Apply transform on-the-fly (no preprocessing wait time)
print("Setting up on-the-fly transforms...")
sft_train = dataset["train"].with_transform(format_for_sft)
sft_eval = dataset["test"].with_transform(format_for_sft)

print(f"SFT Train: {len(sft_train)} | SFT Eval: {len(sft_eval)}")

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
    device_map="auto",
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
    output_dir="models/chess-sft-sequences",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=50,
    save_steps=2000,
    eval_steps=1000,
    eval_strategy="steps",
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="chess-sft-sequences",
    remove_unused_columns=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_train,
    eval_dataset=sft_eval,
)

print("\n" + "="*80)
print("Starting SFT training on move sequences...")
print(f"Training examples: {len(sft_train):,}")
print(f"Eval examples: {len(sft_eval):,}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print("="*80 + "\n")

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
