import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_SERVICE"] = "true"

import random

from transformers import AutoTokenizer

import torch
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from src.data import load_chess_dataset
from src.tokenizer_utils import ensure_chat_template
from src.rewards import (
    format_reward_func,
    legality_reward_func,
    correctness_reward_func,
    stockfish_eval_reward_func,
)

model_name = "unsloth/granite-4.0-h-1b-base-unsloth-bnb-4bit"
model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct"
model_name = "models/chess-sft-warmup-merged"
# NOTE: If you see zero loss, the model doesn't understand the output format yet.
# Run train_sft_warmup.py FIRST to teach the format, then use:
# model_name = "models/chess-sft-warmup"

name_used = "rationale"
rationale_tag = f"<{name_used}>"
move_tag = "<uci_move>"
close_rationale_tag = f"</{name_used}>"
close_move_tag = "</uci_move>"

tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
tokenizer = ensure_chat_template(tokenizer)

train_dataset, test_dataset = load_chess_dataset(tokenizer)

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# Test samples
correct_sample = f"""{rationale_tag}I see that the bishop on f2 can capture the rook on g3, winning material.{close_rationale_tag}
{move_tag}f2g3{close_move_tag}"""

wrong_format = """The best move is f2g3 because it wins the rook."""

illegal_move = f"""{rationale_tag}Moving the pawn forward.{close_rationale_tag}
{move_tag}z9z9{close_move_tag}"""

# Test
test_completions = [correct_sample, wrong_format, illegal_move]
test_legal_moves = [["f2g3", "a8g8", "e7e8"]] * 3
test_correct = ["f2g3"] * 3

print("Format rewards:", format_reward_func(test_completions))
print(
    "Legality rewards:",
    legality_reward_func(test_completions, legal_moves=test_legal_moves),
)
print(
    "Correctness rewards:",
    correctness_reward_func(test_completions, correct_move=test_correct),
)


# GRPO Training config
training_args = GRPOConfig(
    output_dir="models/chess-grpo-qwen3-challenge",
    learning_rate=5e-6,  # Lower LR for stability
    lr_scheduler_type="cosine",
    logging_steps=20,
    max_steps=3000,
    per_device_train_batch_size=4,  # Smaller batch for more diverse samples
    gradient_accumulation_steps=4,  # Keep same effective batch
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific
    max_completion_length=128,  # Reduced - need ~50 tokens max for answer
    num_generations=16,
    beta=0.01,  # Higher KL penalty to stay close to base model
    top_k=50,  # More diverse sampling
    top_p=0.95,
    temperature=1.0,  # Higher temperature for more exploration
    # Logging
    report_to="wandb",
    logging_dir="./logs",
    save_steps=300,
    eval_steps=600,
    run_name="chess-grpo-qwen3-challenge",
)

print("Config ready!")

# Create LoRA config with target modules
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ## 6. Create Trainer and Start Training

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[format_reward_func, legality_reward_func, stockfish_eval_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
)

print("Trainer created successfully!")

# Start training
trainer.train()

# Save the adapter
adapter_path = training_args.output_dir + "/adapter"
trainer.model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# Merge adapter with base model and save full model
print("Merging adapter with base model...")
model = trainer.model.merge_and_unload()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
print(f"Full model saved to {training_args.output_dir}")

# ## 7. Test the Trained Model

# Test on a few puzzles
# Load the trained model
model = AutoModelForCausalLM.from_pretrained(
    training_args.output_dir, dtype=torch.bfloat16, device_map="auto"
)

# Test on 3 puzzles
for i in range(3):
    index = random.randint(0, len(test_dataset) - 1)
    puzzle = test_dataset[index]
    prompt = puzzle["prompt"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.7, do_sample=True
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    print(f"\n{'='*80}")
    print(f"Puzzle {i+1} (Rating: {puzzle['rating']})")
    print(f"Correct move: {puzzle['correct_move']}")
    print(f"{'='*80}")
    print(response)
    print()
