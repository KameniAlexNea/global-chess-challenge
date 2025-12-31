"""
GRPO training on move sequences with separate rewards for rationale and move quality.
Uses Stockfish evaluation lines to teach both tactical reasoning and move selection.
"""

import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random

import torch
from transformers import GenerationConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import prepare_model_for_kbit_training

from src.config import (
    close_move_tag,
    close_rationale_tag,
    move_tag,
    rationale_tag,
)
from src.data import load_grpo_sequences_dataset
from src.rewards import (
    rationale_format_reward_func,
    move_format_reward_func,
    rationale_length_reward_func,
    legality_reward_func,
    stockfish_eval_reward_func,
)
from src.tokenizer_utils import ensure_chat_template

# Model selection
model_name = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"  # Start from SFT checkpoint
tokenizer_name = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True)
tokenizer = ensure_chat_template(tokenizer)

# Load sequences dataset
train_dataset, test_dataset = load_grpo_sequences_dataset(
    tokenizer,
    data_file="data/processed/move_sequences_500mb.jsonl",
    train_samples=50_000,
    num_proc=8,
)

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# Test reward functions with sample completions
print("\n" + "=" * 80)
print("Testing reward functions...")
print("=" * 80)

correct_sample = f"""{rationale_tag}The bishop on f2 can capture the rook on g3, winning material.{close_rationale_tag}
{move_tag}f2g3{close_move_tag}"""

no_rationale = f"""{move_tag}f2g3{close_move_tag}"""

verbose_rationale = f"""{rationale_tag}First, I need to analyze the position carefully. The bishop on f2 is well-placed. Looking at the tactical opportunities, I can see that capturing the rook would be beneficial. After careful consideration of all alternatives, I believe this is the best move because it wins material and improves our position significantly.{close_rationale_tag}
{move_tag}f2g3{close_move_tag}"""

illegal_move = f"""{rationale_tag}Moving the pawn forward.{close_rationale_tag}
{move_tag}z9z9{close_move_tag}"""

test_completions = [correct_sample, no_rationale, verbose_rationale, illegal_move]
test_legal_moves = [["f2g3", "a8g8", "e7e8"]] * 4
test_correct = ["f2g3"] * 4

print("\nTest completions:")
print("1. Correct (concise rationale + move)")
print("2. No rationale (only move)")
print("3. Verbose rationale")
print("4. Illegal move")
print()

print("Rationale format rewards:", rationale_format_reward_func(test_completions))
print("Move format rewards:", move_format_reward_func(test_completions))
print("Rationale length rewards:", rationale_length_reward_func(test_completions))
print(
    "Legality rewards:",
    legality_reward_func(test_completions, legal_moves=test_legal_moves),
)
print()

# Load model
print("Loading model...")
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# GRPO Training config
training_args = GRPOConfig(
    output_dir="models/chess-grpo-sequences",
    learning_rate=5e-6,  # Lower LR for stability
    lr_scheduler_type="cosine",
    logging_steps=20,
    max_steps=5000,
    per_device_train_batch_size=8,  # Reduced for faster iterations
    gradient_accumulation_steps=8,  # Maintain effective batch size
    bf16=True,
    # GRPO specific
    max_completion_length=64,  # Shorter for speed (rationale + move)
    num_generations=4,  # Reduced from 8 for 2x speed boost
    beta=0.01,  # KL penalty
    top_k=64,
    top_p=0.95,
    temperature=1.0,
    generation_kwargs={
        "max_length": 1024,
        "max_new_tokens": 64,  # Match max_completion_length
        "max_time": 30,  # Reduced timeout
        # "length_penalty": 0.5,
    },
    # Logging
    report_to="wandb",
    logging_dir="./logs",
    save_steps=500,
    eval_steps=1000,
    run_name="chess-grpo-sequences-v1",
)

print("\nTraining configuration:")
print(f"  Model: {model_name}")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Generations per prompt: {training_args.num_generations}")

# LoRA config
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

# Create trainer with all reward functions
print("\n" + "=" * 80)
print("Creating GRPO Trainer...")
print("Reward functions:")
print("  1. Rationale format (has <rationale> tags)")
print("  2. Move format (has <uci_move> tags)")
print("  3. Rationale length (concise, one sentence)")
print("  4. Legality (move is legal)")
print("  5. Stockfish eval (move quality vs best move)")
print("=" * 80 + "\n")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        rationale_format_reward_func,
        move_format_reward_func,
        rationale_length_reward_func,
        legality_reward_func,
        stockfish_eval_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
)

print("Trainer created successfully!")

# Start training
print("\n" + "=" * 80)
print("Starting GRPO training...")
print("=" * 80 + "\n")

trainer.train()

# Save adapter
adapter_path = training_args.output_dir + "/adapter"
trainer.model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# Merge and save full model
print("\nMerging adapter with base model...")
model = trainer.model.merge_and_unload()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
print(f"Full model saved to {training_args.output_dir}")

# Test the trained model
print("\n" + "=" * 80)
print("Testing trained model on random examples...")
print("=" * 80 + "\n")

model = AutoModelForCausalLM.from_pretrained(
    training_args.output_dir, dtype=torch.bfloat16, device_map="auto"
)

for i in range(3):
    index = random.randint(0, len(test_dataset) - 1)
    example = test_dataset[index]
    prompt = example["prompt"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    print(f"\nExample {i+1}")
    print(f"Correct move: {example['correct_move']}")
    print(f"Response:\n{response}")
    print("-" * 80)

print("\nTraining complete!")
