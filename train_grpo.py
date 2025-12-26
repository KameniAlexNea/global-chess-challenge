# %% [markdown]
# ## 1. Setup Environment

# %%
# Install required packages
# !pip install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
# !pip install -q "transformers==4.48.1" "datasets==3.1.0" "accelerate==1.3.0" "trl==0.14.0"
# !pip install -q "peft==0.14.0" "bitsandbytes==0.45.0" "python-chess"

# %%
# from huggingface_hub import login

# Login to Hugging Face
# login(token="", add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# %% [markdown]
# ## 2. Load and Prepare Chess Puzzle Dataset

# %%
import json
from datasets import Dataset
from transformers import AutoTokenizer

model_name = "unsloth/Qwen3-4B-Instruct-2507"

# Load your extracted puzzles
puzzle_file = "data/processed/chess_puzzles_231961.jsonl"

puzzles = []
with open(puzzle_file, 'r') as f:
    for line in f:
        puzzles.append(json.loads(line))

print(f"Loaded {len(puzzles)} chess puzzle positions")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample puzzle
print("\nSample puzzle:")
print(json.dumps(puzzles[0], indent=2))

# %% [markdown]
# ## 3. Create Prompt Format for Chess
# 
# We'll use the same format as your prompt.py: `<rationale>` and `<uci_move>` tags.

# %%
def format_chess_prompt(puzzle_data):
    """
    Format a chess puzzle into GRPO training format.
    Model must infer tactical patterns from position alone.
    """
    inp = puzzle_data['input']
    
    # System message
    system_msg = "You are a chess expert. Analyze positions carefully and find the best tactical move."
    
    # User prompt - NO themes (not available during inference)
    user_msg = f"""Analyze this chess position and find the BEST move.

Position (FEN): {inp['fen']}
Side to move: {inp['side_to_move']}
Legal moves: {' '.join(inp['legal_moves'])}

Explain your reasoning in <rationale> tags, then provide the move in <uci_move> tags.
Example: <rationale>Fork attacking king and queen</rationale><uci_move>f2g3</uci_move>"""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    return {
        "prompt": tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        ),
        "correct_move": puzzle_data['output']['move'],
        "legal_moves": inp['legal_moves'],
        "puzzle_id": puzzle_data['metadata']['puzzle_id'],
        "rating": puzzle_data['metadata']['rating']
    }

# Test the formatting
sample = format_chess_prompt(puzzles[0])
print("Sample prompt:")
print(sample['prompt'])
print(f"\nCorrect move: {sample['correct_move']}")

# %%
# Convert all puzzles to GRPO format
formatted_puzzles = [format_chess_prompt(p) for p in puzzles]

# Create Hugging Face dataset
dataset = Dataset.from_list(formatted_puzzles)

# Split train/test
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# %% [markdown]
# ## 4. Define Reward Functions
# 
# We'll create 3 reward functions:
# 1. **Format Reward**: Correct XML tags `<rationale>...</rationale><uci_move>...</uci_move>`
# 2. **Legality Reward**: Move is in the legal moves list
# 3. **Correctness Reward**: Move matches the puzzle solution

# %%
import re

def format_reward_func(completions, **kwargs):
    """
    Reward for correct format: <rationale>...</rationale><uci_move>...</uci_move>
    """
    rewards = []
    
    for completion in completions:
        try:
            # Check format
            regex = r"<rationale>([^<]*(?:<(?!/?rationale>)[^<]*)*)</rationale>\s*<uci_move>([^<]+)</uci_move>"
            match = re.search(regex, completion, re.DOTALL)
            
            if match and len(match.groups()) == 2:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    
    return rewards


def legality_reward_func(completions, legal_moves, **kwargs):
    """
    Reward if the move is legal.
    """
    rewards = []
    
    for completion, legal in zip(completions, legal_moves):
        try:
            # Extract move
            match = re.search(r"<uci_move>([^<]+)</uci_move>", completion)
            if not match:
                rewards.append(0.0)
                continue
            
            move = match.group(1).strip()
            
            # Check if legal
            if move in legal:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)  # Penalize illegal moves
        except:
            rewards.append(0.0)
    
    return rewards


def correctness_reward_func(completions, correct_move, **kwargs):
    """
    Reward if the move is correct (matches puzzle solution).
    """
    rewards = []
    
    for completion, correct in zip(completions, correct_move):
        try:
            # Extract move
            match = re.search(r"<uci_move>([^<]+)</uci_move>", completion)
            if not match:
                rewards.append(-0.5)
                continue
            
            move = match.group(1).strip()
            
            # Check correctness
            if move == correct:
                rewards.append(3.0)  # High reward for correct solution
            else:
                rewards.append(-0.5)  # Small penalty for wrong move
        except:
            rewards.append(-0.5)
    
    return rewards

# %% [markdown]
# ### Test Reward Functions

# %%
# Test samples
correct_sample = """<rationale>I see that the bishop on f2 can capture the rook on g3, winning material.</rationale>
<uci_move>f2g3</uci_move>"""

wrong_format = """The best move is f2g3 because it wins the rook."""

illegal_move = """<rationale>Moving the pawn forward.</rationale>
<uci_move>z9z9</uci_move>"""

# Test
test_completions = [correct_sample, wrong_format, illegal_move]
test_legal_moves = [["f2g3", "a8g8", "e7e8"]] * 3
test_correct = ["f2g3"] * 3

print("Format rewards:", format_reward_func(test_completions))
print("Legality rewards:", legality_reward_func(test_completions, legal_moves=test_legal_moves))
print("Correctness rewards:", correctness_reward_func(test_completions, correct_move=test_correct))

# %% [markdown]
# ## 5. Setup GRPO Training

# %%
from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config

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
    output_dir="models/chess-grpo-qwen3",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=25000,  # Start small for testing
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific
    max_completion_length=256,
    num_generations=8,  # Generate 8 solutions per puzzle
    beta=0.001,  # KL coefficient
    top_k=30,
    
    # Logging
    report_to="wandb",
    logging_dir="./logs",
)

print("Config ready!")

# %% [markdown]
# ## 6. Create Trainer and Start Training

# %%
trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[
        format_reward_func,
        legality_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)

print("Trainer created successfully!")

# %%
# Start training
trainer.train()

# Save the model
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")

# %% [markdown]
# ## 7. Test the Trained Model

# %%
# Test on a few puzzles
import torch
from transformers import AutoModelForCausalLM

# Load the trained model
model = AutoModelForCausalLM.from_pretrained(
    training_args.output_dir,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Test on 3 puzzles
for i in range(3):
    puzzle = puzzles[i]
    prompt = format_chess_prompt(puzzle)['prompt']
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"\n{'='*80}")
    print(f"Puzzle {i+1} (Rating: {puzzle['metadata']['rating']})")
    print(f"Correct move: {puzzle['output']['move']}")
    print(f"{'='*80}")
    print(response)
    print()


