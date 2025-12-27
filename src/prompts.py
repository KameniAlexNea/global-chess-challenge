from src.config import (
    rationale_tag,
    close_rationale_tag,
    move_tag,
    close_move_tag,
)

system_msg = "You are a chess expert. Analyze positions carefully and find the best tactical move from legal moves."

user_msg = f"""Analyze this chess position and find the BEST move.

Position (FEN): 
{{FEN}}

Briefly, FEN describes chess pieces by single letters [PNBRKQ] for white and [pnbrkq] for black. The pieces found in each rank are specified, starting at the top of the board (a8..h8) and describing all eight ranks. Within each rank, all 8 positions must be specified, with one or more empty squares noted with a digit [1..8]. For example, /8/ is an empty rank (no pieces), while /4P3/ specifies four empty squares, a white pawn, and three more empty squares.

Side to move: 
{{side_to_move}}
Legal moves: 
{{legal_moves_uci}}

Your task is to select the best move from legal moves for the side to move.

Explain in one *sentence* your reasoning in {rationale_tag} tags, then provide the move in {move_tag} tags.
Example: {rationale_tag}Fork attacking king and queen{close_rationale_tag}{move_tag}f2g3{close_move_tag}
"""
