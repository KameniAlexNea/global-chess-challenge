#!/usr/bin/env python3
"""Quick test of the Stockfish-based reward function."""

from src.config import (
    close_move_tag,
    close_rationale_tag,
    move_tag,
    rationale_tag,
)
from src.rewards import correctness_reward_func, stockfish_eval_reward_func

# Test position: r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24
# Correct move: f2g3 (puzzle solution)

fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
legal_moves = ["f2g3", "a8g8", "e7e8", "e7e6", "b2e5"]
correct_move = "f2g3"

# Test completions
correct_completion = f"""{rationale_tag}The bishop captures the rook, winning material.{close_rationale_tag}
{move_tag}f2g3{close_move_tag}"""

good_alternative = f"""{rationale_tag}The rook moves to safety.{close_rationale_tag}
{move_tag}e7e6{close_move_tag}"""

mediocre_move = f"""{rationale_tag}The queen attacks.{close_rationale_tag}
{move_tag}b2e5{close_move_tag}"""

completions = [correct_completion, good_alternative, mediocre_move]

# Test both reward functions
print("=" * 80)
print("Testing Stockfish-based vs Binary Correctness Rewards")
print("=" * 80)
print(f"\nPosition: {fen}")
print(f"Correct move: {correct_move}")
print(f"Legal moves: {legal_moves}")

binary_rewards = correctness_reward_func(completions, correct_move=[correct_move] * 3)

stockfish_rewards = stockfish_eval_reward_func(
    completions,
    correct_move=[correct_move] * 3,
    fen=[fen] * 3,
    legal_moves=[legal_moves] * 3,
    depth=12,
)

print("\n" + "=" * 80)
print("Results:")
print("=" * 80)
for i, (comp, bin_rew, stock_rew) in enumerate(
    zip(completions, binary_rewards, stockfish_rewards)
):
    move = comp.split(move_tag)[1].split(close_move_tag)[0]
    print(f"\nMove {i+1}: {move}")
    print(f"  Binary Reward:     {bin_rew:+.2f}")
    print(f"  Stockfish Reward:  {stock_rew:+.2f}")
    if move == correct_move:
        print(f"  Status: ✓ Correct solution")
    else:
        diff = stock_rew - bin_rew
        if diff > 0:
            print(f"  Status: Good alternative (+{diff:.2f} extra reward)")
        else:
            print(f"  Status: Suboptimal move")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("Binary reward:    Only rewards exact puzzle solution")
print("Stockfish reward: Rewards good moves proportionally to their quality")
print("\nThis allows the model to learn from strong alternatives,")
print("not just memorize puzzle solutions!")
