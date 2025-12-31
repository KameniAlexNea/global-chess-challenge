# GRPO Training on SFT Sequences - Implementation Summary

## Overview
Successfully implemented GRPO training using the SFT move sequences dataset with separate reward functions for rationale quality and move selection.

## Changes Made

### 1. Independent XML Extraction (`src/utils.py`)
Added two new functions that extract tags independently:
- `extract_rationale()` - Extracts `<rationale>...</rationale>` content
- `extract_move()` - Extracts `<uci_move>...</uci_move>` content

These work independently so partial outputs still get appropriate rewards.

### 2. New Reward Functions (`src/rewards.py`)
Created 5 separate reward functions:

1. **`rationale_format_reward_func`** - Rewards having rationale tags (±1.0)
2. **`move_format_reward_func`** - Rewards having move tags (±1.0)
3. **`rationale_length_reward_func`** - Rewards concise rationales:
   - 1.0: One sentence, < 150 chars
   - 0.5: Two sentences, < 200 chars
   - -0.5: Too verbose
4. **`legality_reward_func`** - Updated to use independent extraction:
   - 1.0: Legal move
   - -2.0: Illegal move
   - -1.0: No move extracted
5. **`stockfish_eval_reward_func`** - Updated to use independent extraction:
   - 3.0: Perfect move (≤ 0 cp loss)
   - 2.5-2.0: Excellent (0-75 cp loss)
   - 1.5-0.5: Good to okay (75-300 cp loss)
   - < 0.5: Poor move (> 300 cp loss)
   - -4.0: Illegal move

### 3. GRPO Data Loader (`src/data.py`)
Added `load_grpo_sequences_dataset()`:
- Loads from `move_sequences_500mb.jsonl`
- Randomly samples positions from Stockfish lines
- Formats using `user_msg` template from `prompts.py`
- Returns prompt with FEN, side to move, and legal moves
- Not conversation format - single prompt/response

### 4. Training Script (`train_grpo_sft.py`)
New training script with:
- Model: `models/chess-sft-fullsequences-fast/checkpoint-7000` (start from SFT)
- Dataset: 50k samples from move sequences
- 5 reward functions applied to each generation
- GRPO config:
  - 5000 steps
  - Batch size: 8 × 4 accumulation = 32 effective
  - 8 generations per prompt
  - Temperature: 1.0 for exploration
  - Max completion: 128 tokens (room for rationale + move)
- Saves to `models/chess-grpo-sequences/`

### 5. Test Script (`test_grpo_setup.py`)
Validation script that tests:
- XML extraction functions
- All 5 reward functions
- Data loading
- Sample outputs

## Test Results
✅ All extraction functions work correctly
✅ Reward functions return expected values:
- Correct format: [1.0, 0.0, 1.0, 1.0, 0.0]
- Move format: [1.0, 1.0, 1.0, 1.0, 0.0]
- Rationale length: [1.0, -0.5, -0.5, 1.0, -0.5]
- Legality: [1.0, 1.0, 1.0, -2.0, -1.0]
- Stockfish: [3.0, 3.0, 3.0, -4.0, -3.0]
✅ Data loader successfully formats sequences as prompts

## Training Command
```bash
python train_grpo_sft.py
```

Or with GPU selection:
```bash
CUDA_VISIBLE_DEVICES=0 python train_grpo_sft.py
```

## Expected Behavior
The model will learn to:
1. Always use proper XML format (both tags)
2. Write concise one-sentence rationales
3. Only output legal moves
4. Prefer moves with better Stockfish evaluation
5. Balance exploration (temperature=1.0) with quality

## Key Differences from Previous GRPO Training
- **Data source**: Move sequences (Stockfish lines) instead of puzzles
- **Format**: Single prompt instead of puzzle-specific format
- **Rewards**: 5 separate rewards instead of 3
- **Rationale focus**: Explicit reward for concise explanations
- **Starting point**: Fine-tuned SFT model instead of base model

## Next Steps
1. Run `python test_grpo_setup.py` to verify setup
2. Start training with `python train_grpo_sft.py`
3. Monitor WandB for:
   - Reward trends (all 5 should increase)
   - KL divergence (should stay controlled)
   - Sample outputs (check format compliance)
4. Evaluate checkpoints at 1000, 2500, 5000 steps
5. Test final model on competition environment
