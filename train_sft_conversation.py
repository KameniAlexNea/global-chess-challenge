"""
FASTER SFT training - no gradient checkpointing, larger batches.
"""

import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from src.tokenizer_utils import ensure_chat_template


def _get_device_map_for_kbit_training() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def _require_ddp_if_multi_gpu() -> None:
    if not torch.cuda.is_available():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() > 1 and world_size == 1:
        raise RuntimeError(
            "Multiple GPUs detected but distributed training is not initialized. "
            "Do NOT run this with plain `python` (it can trigger torch.nn.DataParallel and crash with bnb 4-bit). "
            "Launch with one process per GPU, e.g. `accelerate launch --num_processes 2 train_sft_conversation.py` "
            "or `torchrun --nproc_per_node=2 train_sft_conversation.py`."
        )


def _wait_for_everyone() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


NAME = "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit"


def main() -> None:
    model_name = NAME

    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    tokenizer = ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    sft_train, sft_eval = load_sft_sequences_dataset(
        tokenizer=tokenizer,
        data_file="data/processed/move_sequences_500mb.jsonl",
        train_samples=1_000_000,
        test_size=0.01,
        max_length=1024,
        num_proc=16,
        seed=42,
        max_user_moves=2,
    )

    # Load model
    print("Loading model...")
    _require_ddp_if_multi_gpu()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=_get_device_map_for_kbit_training(),
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
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        bf16=True,
        gradient_checkpointing=False,
        report_to="wandb",
        run_name="chess-sft-conversation",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        torch_compile=True
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

    _wait_for_everyone()
    if not trainer.is_world_process_zero():
        return

    # Save adapter
    adapter_path = training_args.output_dir + "/adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Merge and save full model
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
