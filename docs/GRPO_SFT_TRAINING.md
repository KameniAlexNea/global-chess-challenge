# Training Guide (SFT + GRPO)

This document explains how training works in this repo, end-to-end, using the **current** scripts.

## What the model learns

All training stages teach the model to output two XML-like tags:

1. `<rationale>...</rationale>`
2. `<uci_move>...</uci_move>`

The move must be a valid UCI move from the provided legal move list.

Prompts are built from [src/prompts.py](../src/prompts.py) and contain:
- FEN
- side to move
- legal moves (UCI)

## Setup

This repo is managed with `uv` (see [pyproject.toml](pyproject.toml)).

```bash
uv sync
```

Stockfish is required for GRPO training (move-quality reward):

```bash
sudo apt-get update
sudo apt-get install -y stockfish
```

## Datasets used

All training in this repo uses the move sequence dataset:

- `data/processed/move_sequences_500mb.jsonl` extracted from [lichess](https://database.lichess.org/#evals)

Each row contains at least:
- `fen`: starting position
- `line`: a space-separated list of UCI moves (a strong line, e.g. Stockfish PV)

The training scripts sample random split points inside each line to create many position->move examples.

## Stage 1: SFT (single move + PV line)

Goal: from a position, predict the next move and also output a short PV continuation.

Target format (assistant output):

`<rationale>{PV moves in UCI}</rationale><uci_move>{next move}</uci_move>`

Important constraint: the move inside `<uci_move>` must be the **first** move of the PV in `<rationale>`.

Implementation:
- Dataset builder: [src/sft_data.py](../src/sft_data.py) `load_sft_single_move_dataset()`
- Training script: [train_sft_single.py](../train_sft_single.py)

Run (1 GPU):

```bash
uv run accelerate launch --num_processes 1 train_sft_single.py
```

Run (2 GPUs):

```bash
uv run accelerate launch --num_processes 2 train_sft_single.py
```

Outputs:
- adapter: `models/chess-sft-single/adapter/`
- merged model: `models/chess-sft-single/`

## Stage 2: SFT (sequence / multi-turn)

Goal: teach the model to play moves in a multi-turn setting. The dataset contains multiple turns; the model learns to emit `<uci_move>...</uci_move>` at each assistant turn.

Implementation:
- Dataset builder: [src/sft_data.py](../src/sft_data.py) `load_sft_sequences_dataset()`
  - labels are masked so only tokens inside `<uci_move>...</uci_move>` are trained
- Training script: [train_sft_conversation.py](../train_sft_conversation.py)

Run (1 GPU):

```bash
uv run accelerate launch --num_processes 1 train_sft_conversation.py
```

Run (2 GPUs):

```bash
uv run accelerate launch --num_processes 2 train_sft_conversation.py
```

Outputs:
- adapter: `models/chess-sft-conversation/adapter/`
- merged model: `models/chess-sft-conversation/`

## Stage 3: GRPO (reinforcement learning on sequences)

Goal: improve move quality while keeping strict formatting.

Implementation:
- Dataset loader: [src/data.py](../src/data.py) `load_grpo_sequences_dataset()`
- Training script: [train_grpo.py](../train_grpo.py)
- Rewards: [src/rewards.py](../src/rewards.py)

The GRPO trainer generates multiple candidate completions per prompt and scores them with reward functions:
- format / tags present
- brevity (token penalty)
- rationale conciseness
- legality (move must be legal)
- Stockfish-based move quality (dominant signal)

Run (1 GPU):

```bash
uv run accelerate launch --num_processes 1 train_grpo.py
```

Run (2 GPUs):

```bash
uv run accelerate launch --num_processes 2 train_grpo.py
```

Outputs:
- adapter: `models/chess-grpo-sequences/adapter/`
- merged model: `models/chess-grpo-sequences/`

## Running a trained model

See the repo-level README for copy/paste commands:
- Quick local generation with Transformers
- vLLM server + `global-chess-challenge-2025-starter-kit/local_evaluation.py`
