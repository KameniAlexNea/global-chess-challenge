"""
GRPO training on move sequences with separate rewards for rationale and move quality.
Uses Stockfish evaluation lines to teach both tactical reasoning and move selection.
"""

import os

os.environ["WANDB_PROJECT"] = "global-chess-challenge"
os.environ["WANDB_WATCH"] = "none"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
from functools import wraps

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from src.data import load_grpo_sequences_dataset
from src.rewards import (
    combined_format_reward_func,
    legality_reward_func,
    rationale_quality_reward_func,
    stockfish_eval_reward_func,
)
from src.tokenizer_utils import ensure_chat_template

_STOCKFISH_DEPTH = 3


def _parse_depth_schedule(spec: str) -> list[tuple[int, int]]:
    """Parse a schedule like '0:1,500:2,1500:3' into sorted (step, depth)."""
    spec = (spec or "").strip()
    if not spec:
        return [(0, 3)]

    items: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step_s, depth_s = part.split(":", 1)
        step = int(step_s.strip())
        depth = int(depth_s.strip())
        if step < 0 or depth <= 0:
            raise ValueError(f"Invalid schedule entry: {part}")
        items.append((step, depth))

    items.sort(key=lambda x: x[0])
    if items[0][0] != 0:
        items.insert(0, (0, items[0][1]))
    return items


def _depth_for_step(step: int, schedule: list[tuple[int, int]]) -> int:
    depth = schedule[0][1]
    for s, d in schedule:
        if step >= s:
            depth = d
        else:
            break
    return depth


class _StockfishDepthSchedulerCallback(TrainerCallback):
    def __init__(self, schedule: list[tuple[int, int]]):
        self._schedule = schedule
        self._last_depth: int | None = None

    def on_step_begin(self, args, state, control, **kwargs):
        global _STOCKFISH_DEPTH
        depth = _depth_for_step(int(state.global_step), self._schedule)
        _STOCKFISH_DEPTH = depth
        if self._last_depth != depth and state.is_world_process_zero:
            print(f"[stockfish] depth -> {depth} (step={state.global_step})")
            self._last_depth = depth
        return control


def _require_ddp_if_multi_gpu() -> None:
    if not torch.cuda.is_available():
        return
    # If you start this script with plain `python train_grpo.py` and have 2+ GPUs
    # visible, HF/Accelerate can fall back to single-process DataParallel.
    # bitsandbytes 4-bit + DataParallel is a common source of CUBLAS failures.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() > 1 and world_size == 1:
        raise RuntimeError(
            "Multiple GPUs detected but distributed training is not initialized. "
            "Do NOT run this with plain `python` (it can trigger torch.nn.DataParallel and crash with bnb 4-bit). "
            "Launch with one process per GPU, e.g. `accelerate launch --num_processes 2 train_grpo.py` "
            "or `torchrun --nproc_per_node=2 train_grpo.py`."
        )


def _get_device_map_for_kbit_training() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def _scale_reward_func(reward_func, weight: float, name: str | None = None):
    base_name = name or getattr(reward_func, "__name__", "reward")

    @wraps(reward_func)
    def scaled(*args, **kwargs):
        rewards = reward_func(*args, **kwargs)
        return [float(r) * weight for r in rewards]

    # TRL logs metrics using the reward function name; make it explicit.
    scaled.__name__ = f"{base_name}_x{weight:g}"
    scaled.__qualname__ = scaled.__name__
    return scaled


NAME = "models/chess-sft-conversation/merged-checkpoint-3000"


def main() -> None:
    # Model selection
    model_name = NAME  # Start from SFT checkpoint
    tokenizer_name = NAME

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = ensure_chat_template(tokenizer)

    # Load sequences dataset
    train_dataset, test_dataset = load_grpo_sequences_dataset(
        tokenizer,
        data_file="data/processed/move_sequences_500mb.jsonl",
        train_samples=50_000,
        num_proc=8,
    )

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

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

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # GRPO Training config
    training_args = GRPOConfig(
        output_dir="models/chess-grpo-sequences",
        learning_rate=5e-6,  # Lower LR for stability
        lr_scheduler_type="cosine",
        logging_steps=50,
        max_steps=2500,
        per_device_train_batch_size=24,  # Reduced for faster iterations
        gradient_accumulation_steps=2,  # Maintain effective batch size
        bf16=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # GRPO specific
        max_completion_length=64,  # Shorter for speed (rationale + move)
        num_generations=8,  # Reduced from 8 for 2x speed boost
        beta=0.01,  # KL penalty
        top_k=64,
        top_p=0.95,
        temperature=1.0,
        generation_kwargs={
            "max_length": 1024,
            "max_new_tokens": 64,  # Match max_completion_length
            "max_time": 30,  # Reduced timeout
        },
        # Logging
        report_to="wandb",
        logging_dir="./logs",
        save_steps=500,
        eval_steps=1000,
        run_name="chess-grpo-sequences-v1",
        # auto_find_batch_size=True,
        ddp_find_unused_parameters=False,
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
    print("  1. Combined format (<rationale> + <uci_move>)")
    print("  2. Rationale quality (min length + basic heuristics)")
    print("  3. Legality (move is legal)")
    print("  4. Stockfish eval (move quality vs precomputed best move)")
    print("=" * 80 + "\n")

    # Actual Stockfish depth curriculum over *training steps*.
    # Default matches the intent of the earlier comment: start cheap, then increase.
    schedule_spec = os.environ.get("STOCKFISH_DEPTH_SCHEDULE", "0:1,500:3")
    schedule = _parse_depth_schedule(schedule_spec)
    print(f"Stockfish depth schedule: {schedule_spec}")

    def stockfish_var_eval_reward_func(completions, **kwargs):
        return stockfish_eval_reward_func(completions, depth=_STOCKFISH_DEPTH, **kwargs)

    # Reward weights: make Stockfish the dominant signal while keeping
    # format/legality constraints as shaping rewards.
    stockfish_reward = _scale_reward_func(
        stockfish_var_eval_reward_func, 3.0, name="stockfish_eval"
    )
    combined_format_reward = _scale_reward_func(
        combined_format_reward_func, 0.5, name="combined_format"
    )
    # Encourage a minimally-informative rationale (avoid one-word rationales).
    rationale_quality_reward = _scale_reward_func(
        lambda completions, **kwargs: rationale_quality_reward_func(
            completions, min_chars=40, max_chars=220, **kwargs
        ),
        0.5,
        name="rationale_quality",
    )
    legality_reward = _scale_reward_func(legality_reward_func, 0.5, name="legality")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        callbacks=[_StockfishDepthSchedulerCallback(schedule)],
        reward_funcs=[
            combined_format_reward,
            rationale_quality_reward,
            legality_reward,
            stockfish_reward,
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

    # Sync and only save/merge/test on rank 0.
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        return

    # Save adapter (safe for 4-bit QLoRA training)
    adapter_path = training_args.output_dir + "/adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print("\nAdapter saved.")
    print("To merge into a full bf16 model (recommended), run:")
    print(f"  python push_full_model.py --adapter_dir {adapter_path} --no_push")

    # Test the trained model
    print("\n" + "=" * 80)
    print("Testing trained model on random examples...")
    print("=" * 80 + "\n")

    # Ad-hoc testing: use the trained adapter model directly (no merge required).
    test_model = trainer.model

    for i in range(3):
        index = random.randint(0, len(test_dataset) - 1)
        example = test_dataset[index]
        prompt = example["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt").to(test_model.device)

        with torch.no_grad():
            outputs = test_model.generate(
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


if __name__ == "__main__":
    main()
