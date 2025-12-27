# Stockfish-Based Evaluation Rewards

## Overview

The `stockfish_eval_reward_func` provides **continuous feedback** based on move quality instead of binary correct/incorrect rewards. This helps the model learn from near-optimal moves and understand the gradations in move quality.

## How It Works

1. **Evaluates the position** after both the predicted move and the correct move using Stockfish
2. **Calculates centipawn difference** between the two evaluations
3. **Scales the reward** based on how close the predicted move is to optimal

## Reward Scale

| Move Quality | Centipawn Loss | Reward Range |
|-------------|----------------|--------------|
| Correct/Equivalent | 0-10 cp | +3.0 |
| Very Good Alternative | 10-50 cp | +2.0 to +2.5 |
| Good Alternative | 50-100 cp | +1.0 to +2.0 |
| Decent Move | 100-200 cp | 0.0 to +1.0 |
| Below Average | 200-400 cp | -0.5 to -1.0 |
| Bad Move | >400 cp | -1.0 to -2.0 |

## Usage in Training

### Option 1: Replace Binary Correctness Reward

```python
from src.rewards import (
    format_reward_func,
    legality_reward_func,
    stockfish_eval_reward_func,  # Use this instead of correctness_reward_func
)

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, legality_reward_func, stockfish_eval_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)
```

### Option 2: Use Both (Combined Signal)

```python
trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[
        format_reward_func, 
        legality_reward_func, 
        correctness_reward_func,  # Binary bonus for exact match
        stockfish_eval_reward_func  # Continuous quality feedback
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)
```

## Benefits

1. **Gradual Learning**: Model receives positive feedback for good moves even when not perfect
2. **Better Exploration**: Encourages trying different strong moves instead of only memorizing puzzle solutions
3. **Robust Evaluation**: Works well even when model outputs alternative strong moves

## Performance Considerations

- **Depth**: Default is 10, which balances speed and accuracy. Lower for faster training, higher for more precise evaluation.
- **Caching**: Uses a global Stockfish engine instance to avoid overhead of starting/stopping
- **Fallback**: Automatically falls back to binary reward if Stockfish is not available

## Example Output

```
Binary correctness reward: [-0.5]  # Wrong move
Stockfish-based reward: [1.5]     # But still a decent move!
```

This shows how a move that's not the puzzle solution can still receive positive feedback if it's tactically sound.
