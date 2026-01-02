"""
SFT data loading and preprocessing for chess move sequences.
"""

import random
from functools import lru_cache
from typing import Dict, Literal, Optional

import chess
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from src.config import (
    close_move_tag,
    close_rationale_tag,
    move_tag,
    rationale_tag,
)
from src.prompts import (
    conversation_system_msg,
    conversation_user_msg_after_move,
    conversation_user_msg_first,
    user_msg_pv_line,
)


@lru_cache(maxsize=10000)
def get_board_at_position(fen: str, moves_tuple: tuple) -> chess.Board:
    """Cache board states to avoid recomputing."""
    board = chess.Board(fen)
    for move_uci in moves_tuple:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
    return board.copy()


def create_single_move_example(line_data: Dict, line_k: int = 6) -> Dict:
    """
    Randomly pick a split point in the line and create training example.

        This creates a single Q&A pair: given a position, predict the next best move
        and provide a principal-variation style continuation.

        Target format:
            <rationale>{PV in UCI}</rationale><uci_move>{next_move}</uci_move>
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

    # PV continuation from this position (includes next_move as first move).
    # If line_k <= 0, include the full remaining continuation from the source line.
    if line_k <= 0:
        pv_moves = moves[split_idx:]
        if not pv_moves:
            pv_moves = [next_move]
    else:
        pv_moves = moves[split_idx : split_idx + line_k]
        if not pv_moves:
            pv_moves = [next_move]

    # Get legal moves
    legal_moves = [m.uci() for m in board.legal_moves]
    side_to_move = "White" if board.turn == chess.WHITE else "Black"

    # Create prompt (aligned with src/prompts.py)
    prompt_text = user_msg_pv_line.format(
        FEN=board.fen(),
        side_to_move=side_to_move,
        legal_moves_uci=" ".join(legal_moves),
    )

    # Create response
    pv_text = " ".join(pv_moves)
    response = (
        f"{rationale_tag}{pv_text}{close_rationale_tag}"
        f"{move_tag}{next_move}{close_move_tag}"
    )

    return {"prompt": prompt_text, "response": response}


def create_conversation_example(
    line_data: Dict, max_user_moves: int = 2
) -> Optional[Dict]:
    """
    Create a full multi-turn conversation from a move sequence.

    Format: Each message includes position (FEN) and legal moves.
    Both user and assistant respond with <uci_move>xxxx</uci_move>.
    This gives the model full context at each turn.

    Args:
        line_data: Dict with at least {"fen", "line"}
        max_user_moves: Cap the number of *user moves* included in the conversation.
            User turns are the verbose ones (they include the move + FEN + legal moves),
            so this is a more reliable knob for controlling token length.
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

    starting_legal_moves_uci = " ".join(m.uci() for m in board.legal_moves)
    starting_side_to_move = "White" if board.turn == chess.WHITE else "Black"

    # Single system message explaining the rules and roles.
    # The rest of the conversation alternates user state updates and assistant moves.
    system_message = conversation_system_msg.format(
        assistant_color_name=assistant_color_name,
        user_color_name=user_color_name,
        starting_fen=fen,
        starting_side_to_move=starting_side_to_move,
        starting_legal_moves_uci=starting_legal_moves_uci,
    )

    messages = [{"role": "system", "content": system_message}]
    has_assistant_move = False
    last_user_move = None
    assistant_moves_added = 0
    user_moves_presented = 0

    # Randomly decide whether the first move is produced immediately by the assistant
    # (only possible if it's the assistant's turn in the starting position).
    start_with_assistant = (board.turn == assistant_color) and random.choice(
        [True, False]
    )

    for move_uci in moves:
        current_turn = board.turn

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                break
        except Exception:
            break

        if current_turn == assistant_color:
            # Assistant's turn - show position + legal moves, assistant responds
            current_fen = board.fen()
            legal_moves = [m.uci() for m in board.legal_moves]
            legal_moves_uci = " ".join(legal_moves)

            # If we decided to start with an assistant move, we skip the initial
            # user state message for the very first assistant move only.
            if last_user_move is None and start_with_assistant:
                messages.append(
                    {"role": "assistant", "content": f"<uci_move>{move_uci}</uci_move>"}
                )
                start_with_assistant = False
                assistant_moves_added += 1
            else:
                if last_user_move is None:
                    # Assistant moves first (user provides starting state)
                    user_content = conversation_user_msg_first.format(
                        FEN=current_fen,
                        legal_moves_uci=legal_moves_uci,
                    )
                else:
                    # If we've already included enough user moves, stop before adding
                    # another verbose user state update.
                    if max_user_moves > 0 and user_moves_presented >= max_user_moves:
                        break
                    # User moved, show their move then position
                    user_content = conversation_user_msg_after_move.format(
                        last_user_move=last_user_move,
                        FEN=current_fen,
                        legal_moves_uci=legal_moves_uci,
                    )
                    user_moves_presented += 1

                last_user_move = None
                messages.append({"role": "user", "content": user_content})
                messages.append(
                    {"role": "assistant", "content": f"<uci_move>{move_uci}</uci_move>"}
                )
                assistant_moves_added += 1
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

    return {
        "messages": messages,
        "num_moves": len(moves),
        "assistant_moves": assistant_moves_added,
        "user_moves": user_moves_presented,
    }


def load_sft_text_examples(
    data_file: str = "data/processed/move_sequences_500mb.jsonl",
    num_samples: int = 100,
    mode: Literal["single_move", "conversation"] = "single_move",
    max_user_moves: int = 2,
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
            result = create_conversation_example(
                line_data, max_user_moves=max_user_moves
            )
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
                        "assistant_moves": result.get("assistant_moves", None),
                        "user_moves": result.get("user_moves", None),
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
    line_k: int = 6,
    eval_max_samples: int = 2000,
    num_proc: int = 16,
    seed: int = 42,
):
    """
    Load and preprocess chess move sequences for SFT training (single move prediction).

    Each training example: given a position, predict the next best move.
    Uses random splits so same line generates different examples.
    Trains on the full assistant response (including <rationale> and <uci_move>).

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
        """Format a single example for supervised learning with label masking."""
        line_data = {
            "fen": example["fen"],
            "line": example["line"],
            "depth": example["depth"],
        }

        # Create training example with random split
        training_example = create_single_move_example(line_data, line_k=line_k)

        user_msg = {"role": "user", "content": training_example["prompt"]}

        # Tokenize prompt and response separately.
        # This gives an exact, template-correct boundary index for label masking.
        def _chat_to_ids(messages: list[dict], add_generation_prompt: bool) -> list[int]:
            try:
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
                # HF returns either a list[int] or a BatchEncoding-like dict.
                if isinstance(ids, dict):
                    return list(ids["input_ids"])
                return list(ids)
            except TypeError:
                # Back-compat for tokenizers that don't support tokenize=True.
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
                return tokenizer(text, add_special_tokens=False)["input_ids"]

        prompt_ids = _chat_to_ids([user_msg], add_generation_prompt=True)
        response_ids = tokenizer(training_example["response"], add_special_tokens=False)[
            "input_ids"
        ]

        input_ids = prompt_ids + response_ids

        # Ensure we have a single EOS at the end (if the model uses one).
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
            input_ids.append(eos_id)

        # Right-side truncation (keep the FEN / board state at the start).
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        unpadded_len = len(input_ids)
        attention_mask = [1] * unpadded_len

        # Labels: train only on the assistant response (everything after the prompt).
        labels = [-100] * unpadded_len
        mask_start_idx = min(len(prompt_ids), unpadded_len)
        for pos in range(mask_start_idx, unpadded_len):
            labels[pos] = input_ids[pos]

        # Right padding to max_length.
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        pad_len = max_length - unpadded_len
        if pad_len > 0:
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

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

    # Cap eval set for speed; large eval sets can stall training for a long time.
    if eval_max_samples and len(dataset["test"]) > eval_max_samples:
        dataset["test"] = dataset["test"].select(range(int(eval_max_samples)))
        print(f"Capped eval set to {len(dataset['test'])} samples")

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
    max_user_moves: int = 2,
    num_proc: int = 16,
    seed: int = 42,
):
    """
    Load and preprocess chess move sequences for SFT training with label masking.

    Only tokens between <uci_move> and </uci_move> are trained (labels != -100).

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

    # Get token IDs for "uci_move" - appears in both <uci_move> and </uci_move>
    uci_move_tokens = tokenizer.encode("uci_move", add_special_tokens=False)

    def find_subsequence(seq, subseq):
        """Find all starting positions of subseq in seq."""
        positions = []
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i : i + len(subseq)] == subseq:
                positions.append(i)
        return positions

    def find_move_token_positions(input_ids: list) -> list:
        """Find positions of move tokens (full <uci_move>...</uci_move> tags)."""
        # Find all occurrences of "uci_move" tokens
        occurrences = find_subsequence(input_ids, uci_move_tokens)

        # Pair them up: (open, close), (open, close), ...
        positions = []
        for i in range(0, len(occurrences) - 1, 2):
            # Include full tags: from < before first uci_move to > after second uci_move
            start = occurrences[i] - 1  # Include < before uci_move
            end = (
                occurrences[i + 1] + len(uci_move_tokens) + 1
            )  # Include > after uci_move
            for pos in range(start, end):
                if 0 <= pos < len(input_ids):
                    positions.append(pos)

        return positions

    def format_for_sft(example):
        """Format a single example as a multi-turn conversation for SFT with label masking."""
        line_data = {
            "fen": example["fen"],
            "line": example["line"],
            "depth": example["depth"],
        }

        # Create conversation
        result = create_conversation_example(line_data, max_user_moves=max_user_moves)

        if result is None:
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

        # Tokenize
        full_tokens = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        input_ids = full_tokens["input_ids"]

        # Create labels - only predict move tokens
        labels = [-100] * len(input_ids)
        move_positions = find_move_token_positions(input_ids)
        for pos in move_positions:
            labels[pos] = input_ids[pos]

        return {
            "input_ids": input_ids,
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

    # Set format for PyTorch
    sft_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    sft_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Final: {len(sft_train):,} train | {len(sft_eval):,} eval examples")

    return sft_train, sft_eval
