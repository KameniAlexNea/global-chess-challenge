import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"

import random

from transformers import AutoTokenizer

import torch
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config
from src.data import load_chess_dataset
from src.rewards import (
    format_reward_func,
    legality_reward_func,
    correctness_reward_func,
)

model_name = "alexneakameni/Qwen3-4B-Instruct-2507-chess-grpo"
model_name = "unsloth/Qwen3-4B-Instruct-2507"
# model_name = "unsloth/Qwen3-1.7B"

name_used = "rationale"
rationale_tag = f"<{name_used}>"
move_tag = "<uci_move>"
close_rationale_tag = f"</{name_used}>"
close_move_tag = "</uci_move>"

tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# Model config
model_config = ModelConfig(
    model_name_or_path=model_name,
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    use_peft=True,
    load_in_4bit=True,
)

# GRPO Training config
training_args = GRPOConfig(
    output_dir="models/chess-grpo-qwen3-challenge",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    logging_steps=20,
    max_steps=3000,  # Validation run - check if loss goes down before committing to marathon
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific
    max_completion_length=256,
    num_generations=16,  # Generate 16 solutions per puzzle
    beta=0.001,  # KL coefficient
    top_k=30,
    top_p=0.9,
    temperature=0.7,
    # Logging
    report_to="wandb",
    logging_dir="./logs",
    save_steps=300,
    eval_steps=600,
    run_name="chess-grpo-qwen3-challenge",
)

print("Config ready!")

# ## 6. Create Trainer and Start Training

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, legality_reward_func, correctness_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)

print("Trainer created successfully!")

# Start training
trainer.train()

# Save the model
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")

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
