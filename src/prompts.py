from src.config import (
    close_move_tag,
    close_rationale_tag,
    move_tag,
    rationale_tag,
)

system_msg = "You are a chess expert. Analyze positions carefully and find the best tactical move from legal moves."

user_msg = f"""Internally analyze the given chess position and select the best move for the side to move.

# Context
- **Position (FEN):**
{{FEN}}
- **Side to Move:** (either 'White' or 'Black')
{{side_to_move}}
- **Legal Moves:** (space-separated UCI moves)
{{legal_moves_uci}}

## FEN Explanation
FEN (Forsyth-Edwards Notation) encodes a chess board position using single letters [PNBRKQ] for White pieces and [pnbrkq] for Black pieces. Each rank (row), starting from the top (a8..h8), is specified. All eight squares in a rank are described, with digits [1..8] indicating consecutive empty squares.
- `/8/` denotes a completely empty rank.
- `/4P3/` means four empty squares, then a white pawn, then three more empty squares.

# Task
Select the best move from the provided legal moves list. If no legal moves exist, determine if it's checkmate or stalemate.

1. `{rationale_tag}...{close_rationale_tag}` - A single concise sentence explaining your move choice
2. `{move_tag}xxxx{close_move_tag}` - The selected valid move in UCI format

If no legal moves exist, output only: use 0000 for null moves
`{rationale_tag}no legal moves available{close_rationale_tag}{move_tag}0000{close_move_tag}`

Do NOT include any other text, explanations, or formatting.
"""


# Variant used for SFT where we want the model to emit a short PV line.
# The immediate next move must be duplicated as the first move of the PV.
user_msg_pv_line = f"""Internally analyze the given chess position and select the best move for the side to move.

# Context
- **Position (FEN):**
{{FEN}}
- **Side to Move:** (either 'White' or 'Black')
{{side_to_move}}
- **Legal Moves:** (space-separated UCI moves)
{{legal_moves_uci}}

# Task
Select the best move from the provided legal moves list.

Return exactly:
1. `{rationale_tag}...{close_rationale_tag}` - A short best-line continuation as space-separated UCI moves (a PV). The FIRST move in this PV must be the same as the move you output in `{move_tag}...{close_move_tag}`.
2. `{move_tag}xxxx{close_move_tag}` - The selected valid move in UCI format (the next move to play).

If no legal moves exist, output only:
`{rationale_tag}no legal moves available{close_rationale_tag}{move_tag}0000{close_move_tag}`

Do NOT include any other text, explanations, or formatting.
"""


# Conversation-game prompts used for multi-turn SFT sequences.
# We keep these here so data generation and inference share consistent wording.
conversation_system_msg = f"""You are playing chess as {{assistant_color_name}}. The user is {{user_color_name}}.

The game starts from this position (FEN): {{starting_fen}}

Side to move: {{starting_side_to_move}}
Legal moves (UCI): {{starting_legal_moves_uci}}

Moves are in UCI notation inside {move_tag}{close_move_tag} tags. Each move alternates the board state.
On each of your turns, you will receive the current position (FEN) and the list of legal moves (UCI). Respond with your move in the same format.

Turn format:
- If the user just played, the user message starts with {move_tag}...{close_move_tag} and then provides the resulting Current position (FEN) and Legal moves (UCI).
- If you play first, you may respond immediately after this system message with {move_tag}...{close_move_tag}.
"""


conversation_user_msg_first = """Current position (FEN): {FEN}
Legal moves (UCI): {legal_moves_uci}"""


conversation_user_msg_after_move = f"""{move_tag}{{last_user_move}}{close_move_tag}

Current position (FEN): {{FEN}}
Legal moves (UCI): {{legal_moves_uci}}"""
