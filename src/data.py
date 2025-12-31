import json
import random

import chess
from datasets import load_dataset
from transformers import Qwen2TokenizerFast

from src.prompts import user_msg


def load_chess_dataset(
    tokenizer: Qwen2TokenizerFast,
    train_samples: int = 50000,
    num_proc: int = 8,
):
    # Load your extracted puzzles - proper streaming, not amateur list loading
    # puzzle_file = "data/processed/chess_puzzles_231961.jsonl"
    # dataset = load_dataset("json", data_files=puzzle_file, split="train")
    dataset = load_dataset("alexneakameni/chess-puzzles", split="train")

    print(f"Loaded {len(dataset)} chess puzzle positions")

    # Sample for faster training
    if len(dataset) > train_samples:
        dataset = dataset.shuffle(seed=42).select(range(train_samples))
        print(f"Sampled {train_samples:,} examples for training")

    # Sample puzzle
    print("\nSample puzzle:")
    print(json.dumps(dataset[0], indent=2))

    def format_chess_prompt(example):
        """
        Format a single chess puzzle into GRPO training format.
        Model must infer tactical patterns from position alone.
        """
        inp = example["input"]
        out = example["output"]
        meta = example["metadata"]

        prompt = user_msg.format(
            FEN=inp["fen"],
            side_to_move=inp["side_to_move"],
            legal_moves_uci=" ".join(inp["legal_moves"]),
        )
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": formatted_prompt,
            "correct_move": out["move"],
            "legal_moves": inp["legal_moves"],
            "fen": inp["fen"],
            "puzzle_id": meta["puzzle_id"],
            "rating": meta["rating"],
            "themes": meta["themes"],
        }

    # Test the formatting with a single sample
    sample = format_chess_prompt(dataset[0])
    print("Sample prompt:")
    print(sample["prompt"])
    print(f"\nCorrect move: {sample['correct_move']}")

    dataset = dataset.train_test_split(test_size=0.01, seed=42)

    # Preprocess with .map() - much faster than on-the-fly transforms
    print(f"\nPreprocessing dataset with {num_proc} workers...")
    train_dataset = dataset["train"].map(
        format_chess_prompt,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        desc="Formatting train set",
    )
    eval_dataset = dataset["test"].map(
        format_chess_prompt,
        remove_columns=dataset["test"].column_names,
        num_proc=num_proc,
        desc="Formatting eval set",
    )

    print(
        f"Preprocessed {len(train_dataset):,} train | {len(eval_dataset):,} eval examples"
    )

    return train_dataset, eval_dataset


def load_grpo_sequences_dataset(
    tokenizer: Qwen2TokenizerFast,
    data_file: str = "data/processed/move_sequences_500mb.jsonl",
    train_samples: int = 50000,
    num_proc: int = 8,
):
    """
    Load move sequences for GRPO training.
    Randomly samples positions from Stockfish lines and formats as prompts
    requiring both rationale and move output.
    """
    # Load sequences
    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Loaded {len(dataset)} move sequences")

    # Sample
    if len(dataset) > train_samples:
        dataset = dataset.shuffle(seed=42).select(range(train_samples))
        print(f"Sampled {train_samples:,} examples for training")

    def format_sequence_prompt(example):
        """
        Randomly pick a position from the sequence and format as GRPO prompt.
        """
        fen = example["fen"]
        moves = example["line"].split()

        if len(moves) < 1:
            return None

        # Randomly pick a position in the sequence
        split_idx = random.randint(0, len(moves) - 1)

        # Build board to that position
        board = chess.Board(fen)
        for move_uci in moves[:split_idx]:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            except:
                return None

        # Get next move (the one we want to predict)
        correct_move = moves[split_idx]

        # Get legal moves
        legal_moves = [m.uci() for m in board.legal_moves]
        side_to_move = "White" if board.turn == chess.WHITE else "Black"

        # Format prompt using the template
        prompt = user_msg.format(
            FEN=board.fen(),
            side_to_move=side_to_move,
            legal_moves_uci=" ".join(legal_moves),
        )

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": formatted_prompt,
            "correct_move": correct_move,
            "legal_moves": legal_moves,
            "fen": board.fen(),
        }

    # Test sample
    sample = format_sequence_prompt(dataset[0])
    if sample:
        print("\nSample prompt:")
        print(sample["prompt"][:500] + "...")
        print(f"\nCorrect move: {sample['correct_move']}")

    # Split train/eval
    dataset = dataset.train_test_split(test_size=0.01, seed=42)

    # Preprocess
    print(f"\nPreprocessing dataset with {num_proc} workers...")
    train_dataset = dataset["train"].map(
        format_sequence_prompt,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        desc="Formatting train set",
    )
    eval_dataset = dataset["test"].map(
        format_sequence_prompt,
        remove_columns=dataset["test"].column_names,
        num_proc=num_proc,
        desc="Formatting eval set",
    )

    # Filter None values
    train_dataset = train_dataset.filter(lambda x: x["prompt"] is not None)
    eval_dataset = eval_dataset.filter(lambda x: x["prompt"] is not None)

    print(
        f"Preprocessed {len(train_dataset):,} train | {len(eval_dataset):,} eval examples"
    )

    return train_dataset, eval_dataset
