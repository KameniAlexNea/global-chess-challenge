"""
SFT data loading and preprocessing for chess move sequences.
"""

import random
from functools import lru_cache
from typing import Dict, Literal, Optional

import chess
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


@lru_cache(maxsize=10000)
def get_board_at_position(fen: str, moves_tuple: tuple) -> chess.Board:
    """Cache board states to avoid recomputing."""
    board = chess.Board(fen)
    for move_uci in moves_tuple:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
    return board.copy()


def create_single_move_example(line_data: Dict) -> Dict:
    """
    Randomly pick a split point in the line and create training example.

    This creates a single Q&A pair: given a position, predict the next best move.
    """
    fen = line_data["fen"]
    moves = line_data["line"].split()

    # Randomly pick where to split (we want to predict move at split_idx)
    # Can predict any move from index 0 to len(moves)-1
    split_idx = random.randint(0, len(moves) - 1)

    # Get board at this position (cached)
    board = get_board_at_position(fen, tuple(moves[:split_idx]))

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

    return {"prompt": prompt_text, "response": response}


def create_conversation_example(line_data: Dict) -> Optional[Dict]:
    """
    Create a full multi-turn conversation from a move sequence.

    Format: Each message includes position (FEN) and legal moves.
    Both user and assistant respond with <uci_move>xxxx</uci_move>.
    This gives the model full context at each turn.
    """
    fen = line_data["fen"]
    moves = line_data["line"].split()

    if len(moves) < 2:
        return None  # Need at least 2 moves for a conversation

    board = chess.Board(fen)

    # Randomly decide if assistant plays White or Black
    assistant_is_white = random.choice([True, False])
    assistant_color = chess.WHITE if assistant_is_white else chess.BLACK
    assistant_color_name = "White" if assistant_is_white else "Black"
    user_color_name = "Black" if assistant_is_white else "White"

    # Add system message explaining the game
    system_message = f"""You are playing chess as {assistant_color_name}. The user is {user_color_name}.

Starting position (FEN): {fen}

Moves are in UCI notation inside <uci_move></uci_move> tags. Each move alternates the board state.
You will receive the current position and legal moves, then respond with your move in the same format."""

    # Build conversation
    messages = [{"role": "system", "content": system_message}]
    has_assistant_move = False
    last_user_move = None

    for move_uci in moves:
        current_turn = board.turn

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                break
        except:
            break

        if current_turn == assistant_color:
            # Assistant's turn - show position + legal moves, assistant responds
            current_fen = board.fen()
            legal_moves = [m.uci() for m in board.legal_moves]

            if last_user_move is None:
                # Assistant moves first
                user_content = f"""Position: {current_fen}
Legal moves: {' '.join(legal_moves)}"""
            else:
                # User moved, show their move then position
                user_content = f"""<uci_move>{last_user_move}</uci_move>

Position: {current_fen}
Legal moves: {' '.join(legal_moves)}"""

            last_user_move = None
            messages.append({"role": "user", "content": user_content})
            messages.append(
                {"role": "assistant", "content": f"<uci_move>{move_uci}</uci_move>"}
            )
            has_assistant_move = True
        else:
            # User's turn - store move to show later
            last_user_move = move_uci

        # Push move to board
        board.push(move)

    if not has_assistant_move or len(messages) < 2:
        return None

    # Ensure conversation ends with assistant
    if messages[-1]["role"] == "user":
        messages = messages[:-1]

    if len(messages) < 2:
        return None

    return {"messages": messages, "num_moves": len(moves)}


def load_sft_text_examples(
    data_file: str = "data/processed/move_sequences_500mb.jsonl",
    num_samples: int = 100,
    mode: Literal["single_move", "conversation"] = "single_move",
    seed: int = 42,
):
    """
    Load chess move sequences and return formatted text examples (no tokenization).

    This allows inspection of the training data before tokenization.

    Args:
        data_file: Path to the JSONL file with move sequences
        num_samples: Number of samples to load
        mode: "single_move" for single Q&A or "conversation" for multi-turn
        seed: Random seed for sampling

    Returns:
        Dataset with formatted text examples
    """
    # Load dataset
    print(f"Loading {num_samples} examples from {data_file}...")
    dataset = load_dataset("json", data_files=data_file)
    full_dataset = dataset["train"]

    # Sample
    if len(full_dataset) > num_samples:
        full_dataset = full_dataset.shuffle(seed=seed).select(range(num_samples))

    examples = []

    for item in full_dataset:
        line_data = {
            "fen": item["fen"],
            "line": item["line"],
            "depth": item["depth"],
        }

        if mode == "single_move":
            # Create single move example
            result = create_single_move_example(line_data)
            examples.append(
                {
                    "type": "single_move",
                    "prompt": result["prompt"],
                    "response": result["response"],
                    "fen": line_data["fen"],
                    "line": line_data["line"],
                    "depth": line_data["depth"],
                }
            )

        elif mode == "conversation":
            # Create conversation example
            result = create_conversation_example(line_data)
            if result is not None:
                # Convert messages to strings for dataset storage
                messages_text = []
                for msg in result["messages"]:
                    messages_text.append(f"{msg['role']}: {msg['content']}")

                examples.append(
                    {
                        "type": "conversation",
                        "messages": result["messages"],
                        "messages_text": "\n\n".join(messages_text),
                        "num_moves": result["num_moves"],
                        "fen": line_data["fen"],
                        "line": line_data["line"],
                        "depth": line_data["depth"],
                    }
                )

        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'single_move' or 'conversation'"
            )

    print(f"Loaded {len(examples)} valid examples")

    # Convert to Dataset
    return Dataset.from_list(examples)


def load_sft_single_move_dataset(
    tokenizer: PreTrainedTokenizer,
    data_file: str = "data/processed/move_sequences_500mb.jsonl",
    train_samples: int = 1_000_000,
    test_size: float = 0.01,
    max_length: int = 512,
    num_proc: int = 16,
    seed: int = 42,
):
    """
    Load and preprocess chess move sequences for SFT training (single move prediction).

    Each training example: given a position, predict the next best move.
    Uses random splits so same line generates different examples.

    Args:
        tokenizer: The tokenizer to use
        data_file: Path to the JSONL file with move sequences
        train_samples: Number of samples to use for training (will sample if more available)
        test_size: Fraction of data to use for evaluation
        max_length: Maximum sequence length for tokenization
        num_proc: Number of parallel workers for preprocessing
        seed: Random seed for sampling and splitting

    Returns:
        Tuple of (train_dataset, eval_dataset) ready for training
    """

    def format_for_sft(example):
        """Format a single example for supervised learning."""
        line_data = {
            "fen": example["fen"],
            "line": example["line"],
            "depth": example["depth"],
        }

        # Create training example with random split
        training_example = create_single_move_example(line_data)

        # Format as chat
        messages = [{"role": "user", "content": training_example["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Full text
        full_text = prompt + training_example["response"] + tokenizer.eos_token

        # Tokenize
        model_inputs = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # Set labels
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    # Load dataset
    print(f"Loading dataset from {data_file}...")
    dataset = load_dataset("json", data_files=data_file)
    full_dataset = dataset["train"]

    # Sample for training
    if len(full_dataset) > train_samples:
        full_dataset = full_dataset.shuffle(seed=seed).select(range(train_samples))
        print(f"Sampled {train_samples:,} examples from full dataset")

    # Train/test split
    dataset = full_dataset.train_test_split(test_size=test_size, seed=seed)
    print(f"Loaded {len(dataset['train'])} train | {len(dataset['test'])} test lines")

    # Preprocess with .map()
    print(f"Preprocessing dataset with {num_proc} workers...")
    sft_train = dataset["train"].map(
        format_for_sft,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        desc="Formatting train set",
    )
    sft_eval = dataset["test"].map(
        format_for_sft,
        remove_columns=dataset["test"].column_names,
        num_proc=num_proc,
        desc="Formatting eval set",
    )

    # Set format for PyTorch
    sft_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    sft_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Final: {len(sft_train):,} train | {len(sft_eval):,} eval examples")

    return sft_train, sft_eval


def load_sft_sequences_dataset(
    tokenizer: PreTrainedTokenizer,
    data_file: str = "data/processed/move_sequences_500mb.jsonl",
    train_samples: int = 1_000_000,
    test_size: float = 0.01,
    max_length: int = 1024,
    num_proc: int = 16,
    seed: int = 42,
):
    """
    Load and preprocess chess move sequences for SFT training.

    Args:
        tokenizer: The tokenizer to use
        data_file: Path to the JSONL file with move sequences
        train_samples: Number of samples to use for training (will sample if more available)
        test_size: Fraction of data to use for evaluation
        max_length: Maximum sequence length for tokenization
        num_proc: Number of parallel workers for preprocessing
        seed: Random seed for sampling and splitting

    Returns:
        Tuple of (train_dataset, eval_dataset) ready for training
    """

    def format_for_sft(example):
        """Format a single example as a multi-turn conversation for SFT."""
        line_data = {
            "fen": example["fen"],
            "line": example["line"],
            "depth": example["depth"],
        }

        # Create conversation
        result = create_conversation_example(line_data)

        if result is None:
            # Return empty/padding for invalid examples
            return {
                "input_ids": [tokenizer.pad_token_id] * max_length,
                "attention_mask": [0] * max_length,
                "labels": [-100] * max_length,
            }

        messages = result["messages"]

        # Apply chat template to full conversation
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize full conversation
        full_tokens = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # Create labels - train on everything (model learns the full pattern)
        labels = full_tokens["input_ids"].copy()

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    # Load dataset
    print(f"Loading dataset from {data_file}...")
    dataset = load_dataset("json", data_files=data_file)
    full_dataset = dataset["train"]

    # Sample for training
    if len(full_dataset) > train_samples:
        full_dataset = full_dataset.shuffle(seed=seed).select(range(train_samples))
        print(f"Sampled {train_samples:,} examples from full dataset")

    # Train/test split
    dataset = full_dataset.train_test_split(test_size=test_size, seed=seed)
    print(f"Loaded {len(dataset['train'])} train | {len(dataset['test'])} test lines")

    # Preprocess with .map()
    print(f"Preprocessing dataset with {num_proc} workers...")
    sft_train = dataset["train"].map(
        format_for_sft,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        desc="Formatting train set",
    )
    sft_eval = dataset["test"].map(
        format_for_sft,
        remove_columns=dataset["test"].column_names,
        num_proc=num_proc,
        desc="Formatting eval set",
    )

    # Filter out invalid examples
    print("Filtering invalid examples...")
    sft_train = sft_train.filter(
        lambda x: sum(x["attention_mask"]) > 10, num_proc=num_proc
    )
    sft_eval = sft_eval.filter(
        lambda x: sum(x["attention_mask"]) > 10, num_proc=num_proc
    )

    # Set format for PyTorch
    sft_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    sft_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Final: {len(sft_train):,} train | {len(sft_eval):,} eval examples")

    return sft_train, sft_eval
