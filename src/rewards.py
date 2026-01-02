import os
import atexit
import math
import re
import shutil
from functools import lru_cache
from typing import Optional

import chess
import chess.engine

from src.utils import extract_move, extract_rationale

# Constants
# NOTE: We keep mate_score=10000 when converting engine scores to centipawns.
# Reward shaping should handle large values smoothly (see log10-based mapping).

# Global Stockfish instance for efficiency
_stockfish_engine: Optional[chess.engine.SimpleEngine] = None
_stockfish_engine_pid: Optional[int] = None


def _close_stockfish_engine() -> None:
    global _stockfish_engine, _stockfish_engine_pid
    if _stockfish_engine is None:
        return
    try:
        _stockfish_engine.quit()
    except Exception:
        try:
            _stockfish_engine.close()
        except Exception:
            pass
    finally:
        _stockfish_engine = None
        _stockfish_engine_pid = None


atexit.register(_close_stockfish_engine)


def get_stockfish_engine():
    """Get or create a global Stockfish engine instance."""
    global _stockfish_engine, _stockfish_engine_pid

    # If we were forked (multiprocessing), don't reuse the parent's engine handle.
    current_pid = os.getpid()
    if _stockfish_engine is not None and _stockfish_engine_pid != current_pid:
        _close_stockfish_engine()

    if _stockfish_engine is None:
        # Prefer env var, then PATH, then common distro path.
        stockfish_path = os.getenv("STOCKFISH_PATH") or shutil.which("stockfish")
        if stockfish_path is None:
            fallback = "/usr/games/stockfish"
            if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
                stockfish_path = fallback

        if stockfish_path is None:
            raise RuntimeError(
                "Stockfish binary not found. Set STOCKFISH_PATH or install stockfish (e.g. apt-get install stockfish)."
            )

        _stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _stockfish_engine.configure({"Threads": 1, "Hash": 64})
        _stockfish_engine_pid = current_pid

    return _stockfish_engine


@lru_cache(maxsize=10000)
def _evaluate_move(position_fen: str, move: str, depth: int) -> int:
    """Cached move evaluation. Returns centipawn score.

    We evaluate the position *after* applying the move (no root_moves constraint).
    The returned value is always from the perspective of the player who played
    `move` in `position_fen`.
    """
    engine = get_stockfish_engine()
    board = chess.Board(position_fen)
    player_color = board.turn
    board.push(chess.Move.from_uci(move))
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    # Make perspective explicit: score from the mover's POV.
    score = info["score"].pov(player_color)
    return score.score(mate_score=10000)


def combined_format_reward_func(completions, **kwargs):
    """
    Combined reward for both rationale and move tags.
    Reduces variance from correlated rewards.
    +2.0 if both present, -1.0 otherwise.
    """
    def _score_move(completion):
        try:
            rationale = extract_rationale(completion)
            move = extract_move(completion)
            if rationale is not None and move is not None:
                return 1.0
            elif rationale is not None or move is not None:
                return 0.0
            else:
                return -1.0
        except:
            return -1.0
    rewards = [
        _score_move(completion)
        for completion in completions
    ]
    return rewards


def rationale_quality_reward_func(
    completions,
    min_chars: int = 40,
    max_chars: int = 220,
    **kwargs,
):
    """Reward for a minimally-informative rationale.

    Goal: prevent degenerate one-word rationales while not incentivizing essays.
    This stays lightweight (no external NLP) and should be used as *shaping* only.

    Heuristics:
    - Missing rationale -> negative
    - Too short (< min_chars) -> negative (scaled)
    - In-range length -> small positive
    - Too long (> max_chars) -> negative
    - 1-2 sentences -> small positive, 0 or many -> negative
    - Small bonus if it mentions common chess concepts (discourages generic filler)
    """
    chess_keywords = (
        "check",
        "mate",
        "fork",
        "pin",
        "skewer",
        "capture",
        "threat",
        "defend",
        "attack",
        "win",
        "queen",
        "rook",
        "bishop",
        "knight",
        "pawn",
        "promotion",
        "material",
        "tempo",
        "tactic",
    )

    keyword_patterns = [re.compile(r"\\b" + re.escape(k) + r"\\b", re.IGNORECASE) for k in chess_keywords]

    rewards = []
    for completion in completions:
        try:
            rationale = extract_rationale(completion)
            if rationale is None:
                rewards.append(-1.0)
                continue

            r = rationale.strip()
            length = len(r)
            if length == 0:
                rewards.append(-1.0)
                continue

            # Sentence-ish count (still heuristic, but less brittle than splitting only on '.')
            sentence_count = len([s for s in re.split(r"[.!?]+", r) if s.strip()])

            reward = 0.0

            # Length shaping (gentler near the threshold)
            if length < min_chars:
                # 0 chars -> -0.6, min_chars -> 0.0
                short_frac = max(0.0, float(min_chars - length) / float(min_chars))
                reward -= 0.6 * short_frac
            elif length <= max_chars:
                reward += 0.3
            else:
                # Penalize rambling
                over = min(1.0, float(length - max_chars) / float(max_chars))
                reward -= 0.3 * over

            # Encourage at least one real sentence, but not a paragraph
            if sentence_count == 1 or sentence_count == 2:
                reward += 0.2
            elif sentence_count == 0 or sentence_count >= 4:
                reward -= 0.2

            # Keyword bonus with word boundaries + diminishing returns.
            found_keywords = 0
            for pat in keyword_patterns:
                if pat.search(r):
                    found_keywords += 1
            keyword_bonus = min(0.2, 0.05 * found_keywords)
            reward += keyword_bonus

            rewards.append(float(reward))
        except Exception:
            rewards.append(-0.5)

    return rewards


def legality_reward_func(completions, legal_moves, **kwargs):
    """
    Reward if the move is legal. Heavy penalty for illegal moves.
    This is a hard constraint - model must learn chess rules first.
    Uses independent move extraction.
    """
    def _score_move(completion, legal):
        try:
            move = extract_move(completion)
            if move is None:
                return -1.0  # Failed to extract any move

            if move in legal:
                return 1.0
            else:
                return -2.0  # Heavy penalty for illegal moves
        except:
            return -1.0
    rewards = [
        _score_move(completion, legal)
        for completion, legal in zip(completions, legal_moves)
    ]
    return rewards


def stockfish_eval_reward_func(
    completions, correct_move, fen, legal_moves, depth=3, **kwargs
):
    """
    Continuous reward based on Stockfish evaluation.
    Compares move quality using centipawn evaluation from current position.
    Uses LRU caching for ~2x speedup on repeated positions.
    Short-circuits early for invalid/illegal moves to avoid expensive Stockfish calls.

    Args:
        depth: Stockfish search depth. Use 1 early in training, 3+ later.
    """
    def _compute_reward(completion, correct, position_fen, legal):
        # Short-circuit: extract move first (cheap)
        move = extract_move(completion)

        if move is None:
            return -3.0

        if move not in legal:
            return -4.0

        # Only call Stockfish for valid moves
        try:

            # Evaluate resulting positions (cached)
            cp_pred = _evaluate_move(position_fen, move, depth)
            cp_correct = _evaluate_move(position_fen, correct, depth)

            # Centipawn loss relative to the reference move
            cp_loss = max(0.0, float(cp_correct - cp_pred))

            # Smooth shaping: penalty grows like log10(cp_loss)
            # - cp_loss=0 => reward=3.0
            # - cp_loss=100 => ~0.0 (with alpha=1.5)
            alpha = 1.5
            reward = 3.0 - alpha * math.log10(1.0 + cp_loss)
            reward = float(max(-2.0, min(3.0, reward)))

            return reward

        except (ValueError, chess.IllegalMoveError, KeyError) as e:
            print(f"Error in stockfish eval: {e}")
            return -1.0
    
    rewards = [
        _compute_reward(completion, correct, position_fen, legal)
        for completion, correct, position_fen, legal in zip(
            completions, correct_move, fen, legal_moves
        )
    ]
    return rewards
