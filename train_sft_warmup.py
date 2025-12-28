"""
Supervised Fine-Tuning warmup to teach the model the output format
before applying GRPO. This is critical - the model must learn to produce
valid XML format before GRPO can provide meaningful gradient signals.
"""

import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_SERVICE"] = "true"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from src.tokenizer_utils import ensure_chat_template
import torch
from transformers import Qwen2TokenizerFast

model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct"

tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
tokenizer = ensure_chat_template(tokenizer)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load raw dataset directly
dataset = load_dataset("alexneakameni/chess-puzzles", split="train")
dataset = dataset.train_test_split(test_size=0.01, seed=42)

print(f"Loaded {len(dataset['train'])} train | {len(dataset['test'])} test puzzles")


def format_for_sft(examples):
    """Format puzzles into supervised learning format with correct answers."""
    inputs = examples["input"]
    outputs = examples["output"]
    metadatas = examples["metadata"]

    texts = []
    for inp, out, meta in zip(inputs, outputs, metadatas):
        # Create prompt using the same format as training
        from src.prompts import user_msg

        prompt_text = user_msg.format(
            FEN=inp["fen"],
            side_to_move=inp["side_to_move"],
            legal_moves_uci=" ".join(inp["legal_moves"]),
        )
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Get themes for rationale
        themes = meta.get("themes", [])
        if themes and len(themes) > 0:
            rationale = " ".join(themes)
        else:
            rationale = "Best tactical move"

        # Create response
        response = (
            f"<rationale>{rationale}</rationale><uci_move>{out['move']}</uci_move>"
        )
        full_text = prompt + response + tokenizer.eos_token
        texts.append(full_text)

    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Set labels (same as input_ids for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    # Mask the prompt part so we only compute loss on the response
    # This is more efficient but optional for now

    return model_inputs


# Format datasets for SFT
sft_train = (
    dataset["train"]
    .select(range(min(10000, len(dataset["train"]))))
    .map(format_for_sft, batched=True, remove_columns=dataset["train"].column_names)
)
sft_eval = (
    dataset["test"]
    .select(range(min(500, len(dataset["test"]))))
    .map(format_for_sft, batched=True, remove_columns=dataset["test"].column_names)
)

print(f"SFT Train: {len(sft_train)} | SFT Eval: {len(sft_eval)}")

# Load model with 4-bit quantization
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

# Prepare model for k-bit training
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

# SFT Training config
training_args = TrainingArguments(
    output_dir="models/chess-sft-warmup",
    num_train_epochs=1,  # Just 1 epoch to learn format
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_steps=20,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="chess-sft-warmup",
    remove_unused_columns=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_train,
    eval_dataset=sft_eval,
)

print("Starting SFT warmup training...")
trainer.train()

# Save the adapter
adapter_path = training_args.output_dir + "/adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# Merge adapter with base model and save full model
print("Merging adapter with base model...")
model = model.merge_and_unload()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
print(f"Full model saved to {training_args.output_dir}")
print(f"Model saved to {training_args.output_dir}")

print("\n" + "=" * 80)
print("SFT warmup complete! Now use this model for GRPO training.")
print("Update train_grpo.py to use: models/chess-sft-warmup")
print("=" * 80)
