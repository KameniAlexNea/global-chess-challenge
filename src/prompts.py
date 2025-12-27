from src.config import (
    rationale_tag,
    close_rationale_tag,
    move_tag,
    close_move_tag,
)

system_msg = "You are a chess expert. Analyze positions carefully and find the best tactical move from legal moves."

user_msg = f"""# Role and Objective
- Analyze a given chess position and recommend the best move for the side to move.

# Instructions
- Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
- Receive input including the FEN string (chess position), the side to move ('White' or 'Black'), and a list of legal moves in UCI format.
- Select the best move from the provided list of legal moves.
- If no legal moves are available, determine if the position is checkmate or stalemate, and explain accordingly without suggesting a move.
- Use a single sentence to briefly explain your choice.
- After selecting or not selecting a move, validate that your output format matches the required tags and that the rationale is both present and concise.

## FEN Explanation
- FEN (Forsyth-Edwards Notation) encodes a chess board position using single letters [PNBRKQ] for White pieces and [pnbrkq] for Black pieces.
- Each rank (row), starting from the top (a8..h8), is specified. All eight squares in a rank are described, with digits [1..8] indicating consecutive empty squares.
- Examples:
  - `/8/` denotes a completely empty rank.
  - `/4P3/` means four empty squares, then a white pawn, then three more empty squares.

## Context Fields
- **Position (FEN):** `{{FEN}}`
- **Side to Move:** `{{side_to_move}}` (either 'White' or 'Black')
- **Legal Moves:** `{{legal_moves_uci}}` (comma-separated UCI moves, e.g., `e2e4,g1f3`)

# Output Format
- Always use `{rationale_tag}...{close_rationale_tag}` to enclose your one-sentence reasoning for move selection.
- Provide the chosen move in `{move_tag}...{close_move_tag}` tags.
- If there are no legal moves, state the reason (checkmate or stalemate) within `{rationale_tag}...{close_rationale_tag}` tags only. Do **not** output a move tag in this case.
  - Example: `{rationale_tag}Checkmate: no legal moves available{close_rationale_tag}`

# Verbosity
- Explanations should be concise (one sentence).

# Stop Conditions
- The response is complete once rationale (and the move, if legal moves exist) is provided per the format above.
"""
