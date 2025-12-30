"""
Test script to verify reward functions and data loading work correctly.
Run this before starting full GRPO training.
"""

import os
os.environ["WANDB_DISABLED"] = "true"

from src.config import close_move_tag, close_rationale_tag, move_tag, rationale_tag
from src.rewards import (
    rationale_format_reward_func,
    move_format_reward_func,
    rationale_length_reward_func,
    legality_reward_func,
    stockfish_eval_reward_func,
)
from src.utils import extract_rationale, extract_move

print("="*80)
print("Testing XML extraction functions")
print("="*80)

# Test extractions
test_text_1 = f"{rationale_tag}Bishop takes rook wins material.{close_rationale_tag}{move_tag}f2g3{close_move_tag}"
test_text_2 = f"{move_tag}f2g3{close_move_tag}"  # No rationale
test_text_3 = f"{rationale_tag}Good move.{close_rationale_tag}"  # No move
test_text_4 = "Plain text with no tags"

print(f"\nText 1: {test_text_1[:50]}...")
print(f"  Rationale: {extract_rationale(test_text_1)}")
print(f"  Move: {extract_move(test_text_1)}")

print(f"\nText 2: {test_text_2}")
print(f"  Rationale: {extract_rationale(test_text_2)}")
print(f"  Move: {extract_move(test_text_2)}")

print(f"\nText 3: {test_text_3}")
print(f"  Rationale: {extract_rationale(test_text_3)}")
print(f"  Move: {extract_move(test_text_3)}")

print(f"\nText 4: {test_text_4}")
print(f"  Rationale: {extract_rationale(test_text_4)}")
print(f"  Move: {extract_move(test_text_4)}")

print("\n" + "="*80)
print("Testing reward functions")
print("="*80)

# Test completions
completions = [
    f"{rationale_tag}The bishop captures the rook, winning material.{close_rationale_tag}{move_tag}f2g3{close_move_tag}",
    f"{move_tag}f2g3{close_move_tag}",  # No rationale
    f"{rationale_tag}First I analyze. The position is complex. Many factors matter. Bishop is strong. Rook can be captured. This seems good. Material advantage. I think this works. Let me explain more. After deep analysis of all possibilities and considering the strategic implications...{close_rationale_tag}{move_tag}f2g3{close_move_tag}",  # Too verbose
    f"{rationale_tag}Pawn forward.{close_rationale_tag}{move_tag}z9z9{close_move_tag}",  # Illegal
    "No tags at all",
]

legal_moves_list = [
    ["f2g3", "a8g8", "e7e8"],
    ["f2g3", "a8g8", "e7e8"],
    ["f2g3", "a8g8", "e7e8"],
    ["f2g3", "a8g8", "e7e8"],
    ["f2g3", "a8g8", "e7e8"],
]

correct_moves = ["f2g3"] * 5
fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 5

print("\nCompletions:")
for i, c in enumerate(completions):
    print(f"  {i+1}. {c[:60]}{'...' if len(c) > 60 else ''}")

print("\nRationale format rewards:", rationale_format_reward_func(completions))
print("Move format rewards:", move_format_reward_func(completions))
print("Rationale length rewards:", rationale_length_reward_func(completions))
print("Legality rewards:", legality_reward_func(completions, legal_moves=legal_moves_list))

print("\nStockfish eval rewards (depth=3):")
try:
    stockfish_rewards = stockfish_eval_reward_func(
        completions, 
        correct_move=correct_moves,
        fen=fens,
        legal_moves=legal_moves_list,
        depth=3
    )
    print(f"  {stockfish_rewards}")
except Exception as e:
    print(f"  Error: {e}")
    print("  (This is expected if Stockfish is not available)")

print("\n" + "="*80)
print("Testing data loader")
print("="*80)

try:
    from transformers import AutoTokenizer
    from src.data import load_grpo_sequences_dataset
    
    tokenizer = AutoTokenizer.from_pretrained("models/chess-grpo-sft-merged", fix_mistral_regex=True)
    
    print("\nLoading small sample (100 examples)...")
    train_ds, test_ds = load_grpo_sequences_dataset(
        tokenizer,
        train_samples=100,
        num_proc=2,
    )
    
    print(f"\nLoaded: {len(train_ds)} train, {len(test_ds)} test")
    
    if len(train_ds) > 0:
        sample = train_ds[0]
        print("\nSample example:")
        print(f"  Prompt length: {len(sample['prompt'])} chars")
        print(f"  Correct move: {sample['correct_move']}")
        print(f"  Legal moves: {len(sample['legal_moves'])} moves")
        print(f"  FEN: {sample['fen'][:50]}...")
        
except Exception as e:
    print(f"Error loading data: {e}")
    print("Make sure data file exists and model tokenizer is available")

print("\n" + "="*80)
print("All tests complete!")
print("="*80)
