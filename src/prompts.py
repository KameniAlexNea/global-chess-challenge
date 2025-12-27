from src.config import (
    rationale_tag,
    close_rationale_tag,
    move_tag,
    close_move_tag,
)

system_msg = "You are a chess expert. Analyze positions carefully and find the best tactical move from legal moves."

user_msg = f"""# Role and Objective
Analyze the given chess position and select the best move for the side to move.

# Context
You will receive:
- **Position (FEN):** `{{FEN}}`
- **Side to Move:** `{{side_to_move}}` (either 'White' or 'Black')
- **Legal Moves:** `{{legal_moves_uci}}` (space-separated UCI moves, e.g., `e2e4 g1f3`)

## FEN Explanation
FEN (Forsyth-Edwards Notation) encodes a chess board position using single letters [PNBRKQ] for White pieces and [pnbrkq] for Black pieces. Each rank (row), starting from the top (a8..h8), is specified. All eight squares in a rank are described, with digits [1..8] indicating consecutive empty squares.
- `/8/` denotes a completely empty rank.
- `/4P3/` means four empty squares, then a white pawn, then three more empty squares.

# Task
Analyze the position considering:
- Piece activity and coordination
- King safety and threats
- Tactical opportunities (checks, captures, forks, pins, skewers)
- Material balance and positional advantages
- Forcing moves and critical responses

Select the best move from the provided legal moves list. If no legal moves exist, determine if it's checkmate or stalemate.

# Output Format (CRITICAL)
Output ONLY the following, with no preamble, checklist, or extra text:

1. `{rationale_tag}...{close_rationale_tag}` - A single concise sentence explaining your move choice
2. `{move_tag}...{close_move_tag}` - The selected move in UCI format

If no legal moves exist, output only:
`{rationale_tag}Checkmate: no legal moves available{close_rationale_tag}`
OR
`{rationale_tag}Stalemate: no legal moves available{close_rationale_tag}`

Do NOT include any other text, explanations, or formatting.
"""
