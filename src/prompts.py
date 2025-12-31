from src.config import (
    close_move_tag,
    close_rationale_tag,
    move_tag,
    rationale_tag,
)

system_msg = "You are a chess expert. Analyze positions carefully and find the best tactical move from legal moves."

user_msg = f"""Internally analyze the given chess position and select the best move for the side to move.

# Context
You will receive:
- **Position (FEN):**
{{FEN}}
- **Side to Move:** (either 'White' or 'Black')
{{side_to_move}}
- **Legal Moves:** (space-separated UCI moves, e.g., `e2e4 g1f3`)
{{legal_moves_uci}}

## FEN Explanation
FEN (Forsyth-Edwards Notation) encodes a chess board position using single letters [PNBRKQ] for White pieces and [pnbrkq] for Black pieces. Each rank (row), starting from the top (a8..h8), is specified. All eight squares in a rank are described, with digits [1..8] indicating consecutive empty squares.
- `/8/` denotes a completely empty rank.
- `/4P3/` means four empty squares, then a white pawn, then three more empty squares.

# Task
Select the best move from the provided legal moves list. If no legal moves exist, determine if it's checkmate or stalemate.

1. `{move_tag}xxxx{close_move_tag}` - The selected 4 digits move in UCI format
2. `{rationale_tag}...{close_rationale_tag}` - A single concise sentence explaining your move choice

If no legal moves exist, output only: use 0000 for null moves
`{move_tag}0000{close_move_tag}{rationale_tag}no legal moves available{close_rationale_tag}`

Do NOT include any other text, explanations, or formatting.
"""
