import os
from functools import lru_cache
from typing import Optional

import chess
import chess.engine

from src.utils import extract_xml_answer, extract_rationale, extract_move

# Constants
MATE_THRESHOLD = 9000  # Centipawn threshold for mate detection (~90 pawns)

# Global Stockfish instance for efficiency
_stockfish_engine: Optional[chess.engine.SimpleEngine] = None


def get_stockfish_engine():
    """Get or create a global Stockfish engine instance."""
    global _stockfish_engine
    if _stockfish_engine is None:
        # Find Stockfish binary
        stockfish_paths = [
            # "/usr/local/bin/stockfish",
            # "/usr/bin/stockfish",
            # "/opt/homebrew/bin/stockfish",
            "/usr/games/stockfish",
        ]
        stockfish_path = None
        for path in stockfish_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                stockfish_path = path
                break

        if stockfish_path is None:
            raise RuntimeError(
                "Stockfish binary not found. Install with: apt-get install stockfish"
            )

        _stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _stockfish_engine.configure({"Threads": 1, "Hash": 64})

    return _stockfish_engine


@lru_cache(maxsize=10000)
def _evaluate_move(position_fen: str, move: str, depth: int) -> int:
    """Cached move evaluation. Returns centipawn score."""
    engine = get_stockfish_engine()
    board = chess.Board(position_fen)
    info = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        root_moves=[chess.Move.from_uci(move)],
    )
    score = info["score"].relative
    return score.score(mate_score=10000)


def format_reward_func(completions, **kwargs):
    """
    Reward for correct XML format: <rationale>...</rationale><uci_move>...</uci_move>
    Strict on format to teach proper structure.
    """
    rewards = []
    for completion in completions:
        try:
            _, _, has_format = extract_xml_answer(completion)
            rewards.append(1.0 if has_format else 0.0)
        except:
            rewards.append(0.0)
    return rewards


def rationale_format_reward_func(completions, **kwargs):
    """
    Reward for having rationale tags, independent of move tags.
    """
    rewards = []
    for completion in completions:
        try:
            rationale = extract_rationale(completion)
            rewards.append(1.0 if rationale is not None else 0.0)
        except:
            rewards.append(0.0)
    return rewards


def move_format_reward_func(completions, **kwargs):
    """
    Reward for having move tags, independent of rationale tags.
    """
    rewards = []
    for completion in completions:
        try:
            move = extract_move(completion)
            rewards.append(1.0 if move is not None else 0.0)
        except:
            rewards.append(0.0)
    return rewards


def rationale_length_reward_func(completions, **kwargs):
    """
    Reward for concise rationale (one sentence, < 150 chars).
    Encourages brief, focused explanations.
    """
    rewards = []
    for completion in completions:
        try:
            rationale = extract_rationale(completion)
            if rationale is None:
                rewards.append(-0.5)
                continue

            # Count sentences (rough heuristic)
            sentence_count = len([s for s in rationale.split(".") if s.strip()])
            length = len(rationale)

            if sentence_count == 1 and length < 150:
                rewards.append(1.0)
            elif sentence_count <= 2 and length < 200:
                rewards.append(0.5)
            else:
                rewards.append(-0.5)  # Penalty for verbose rationales
        except:
            rewards.append(-0.5)
    return rewards


def legality_reward_func(completions, legal_moves, **kwargs):
    """
    Reward if the move is legal. Heavy penalty for illegal moves.
    This is a hard constraint - model must learn chess rules first.
    Uses independent move extraction.
    """
    rewards = []
    for completion, legal in zip(completions, legal_moves):
        try:
            move = extract_move(completion)
            if move is None:
                rewards.append(-1.0)  # Failed to extract any move
                continue

            if move in legal:
                rewards.append(1.0)
            else:
                rewards.append(-2.0)  # Heavy penalty for illegal moves
        except:
            rewards.append(-1.0)
    return rewards


def correctness_reward_func(completions, correct_move, **kwargs):
    """
    Reward if the move is correct (matches puzzle solution).
    Uses independent move extraction.
    """
    rewards = []
    for completion, correct in zip(completions, correct_move):
        try:
            move = extract_move(completion)
            if move is None:
                rewards.append(-0.5)
                continue

            if move == correct:
                rewards.append(3.0)  # High reward for correct solution
            else:
                rewards.append(-0.5)  # Small penalty for wrong move
        except:
            rewards.append(-0.5)
    return rewards


def stockfish_eval_reward_func(
    completions, correct_move, fen, legal_moves, depth=3, **kwargs
):
    """
    Continuous reward based on Stockfish evaluation.
    Compares move quality using centipawn evaluation from current position.
    Uses LRU caching for ~2x speedup on repeated positions.
    Uses independent move extraction.
    """
    rewards = []

    for completion, correct, position_fen, legal in zip(
        completions, correct_move, fen, legal_moves
    ):
        try:
            move = extract_move(completion)

            if move is None:
                rewards.append(-3.0)  # Failed to extract
                continue

            if move not in legal:
                rewards.append(-4.0)  # Illegal move
                continue

            # Use cached evaluation
            cp_pred = _evaluate_move(position_fen, move, depth)
            cp_correct = _evaluate_move(position_fen, correct, depth)

            # Special handling for mate positions
            if abs(cp_correct) >= MATE_THRESHOLD:  # Correct move is mate/getting mated
                if abs(cp_pred) >= MATE_THRESHOLD:  # Predicted is also mate
                    reward = 3.0
                elif cp_pred * cp_correct > 0 and abs(cp_pred) > 200:
                    # Same side, winning position
                    reward = 1.5
                else:
                    reward = -1.0
            else:
                # Normal centipawn comparison
                cp_loss = cp_correct - cp_pred

                # Reward scaling
                if cp_loss <= 0:
                    reward = 3.0
                elif cp_loss < 25:
                    reward = 2.5
                elif cp_loss < 75:
                    reward = 2.0 - (cp_loss - 25) / 50 * 0.5  # 2.0 → 1.5
                elif cp_loss < 150:
                    reward = 1.5 - (cp_loss - 75) / 75 * 1.0  # 1.5 → 0.5
                elif cp_loss < 300:
                    reward = 0.5 - (cp_loss - 150) / 150 * 1.0  # 0.5 → -0.5
                else:
                    # Cap at -2.0 for very bad moves
                    reward = max(-2.0, -0.5 - (cp_loss - 300) / 400)

            rewards.append(reward)

        except (ValueError, chess.IllegalMoveError, KeyError) as e:
            print(f"Error in stockfish eval: {e}")
            rewards.append(-1.0)

    return rewards
