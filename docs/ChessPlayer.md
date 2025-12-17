## Global Chess Challenge

> Most chess players don’t have regular access to a top coach. What they do have are their own games and a recurring question: *“What should I have played here?”* The Global Chess Challenge imagines a tool that looks at those positions, suggests a strong move, and explains the idea in simple language, so players can coach themselves using the games they already play.

Participants build models that play legal chess moves and briefly explain their choices, while a world-class engine checks how well those moves hold up on the board. The challenge turns a familiar game into a testbed to see whether reasoning models can think clearly, play good moves, and talk about them in a way humans can follow.

## 💬 Introduction

Large language models have transformed how systems read and generate text, yet their behaviour in structured domains is still fragile. In games like chess, they can describe ideas in natural language but often overlook simple tactics, mis-handle rules, or fail to follow consistent plans. Classic chess engines calculate and play at superhuman strength, but they do not explain their ideas in simple sentences. The gap between strong play and clear reasoning is what this challenge addresses.

Chess is a useful testing ground for this gap. The rules are fixed, the environment is fully observable, and every move can be checked against a strong engine such as Stockfish. Positions can be encoded as text, moves can be expressed in a standard format, and the quality of each decision can be measured objectively. At the same time, millions of players already use chess tools for training, which makes the setting intuitive for both researchers and practitioners outside machine learning.

The Global Chess Challenge uses this structure to study text-based reasoning: models see only a textual description of the position, must choose a legal move, and provide a short explanation of the idea behind that move, with Stockfish acting as an objective reference.

## 💻 What is the Global Chess Challenge?

The Global Chess Challenge asks participants to build a text-only chess agent that does two things at once: play a legal move and explain the idea behind it in simple language.

On each turn, a model receives a single chess position as text and must respond with a move plus a short rationale. The environment checks that the move parses correctly, is legal in the given position, and can be evaluated by Stockfish for strength.

Participants are free to choose how they implement their agent. They can finetune a language model that reads the prompt and generates a move and rationale, or design a system that combines an LLM with lightweight search or heuristics, as long as the final output respects the required format.

In short, the task is to teach models to make one good, legal move at a time and to explain it clearly, using chess as a clean, verifiable testbed for reasoning.

## 🧪 Tracks and Approaches

The challenge is designed around two complementary research tracks that participants can explore in parallel.

### 1️⃣ Data-centric finetuning

This track focuses on supervised finetuning using open chess data and frames the task as a position-to-move (plus rationale) supervised learning problem on large-scale chess corpora.

Participants can:

* Use positions from the Lichess Open Database and Lichess puzzles (both CC0).
* Parse PGNs and subsample positions across the game (opening, middlegame, endgame), building tuples such as
  `{FEN, side_to_move, legal_moves_uci, move_played, optional Stockfish labels}`.
* Call Stockfish at fixed depth or time to record best moves, principal variation, and evaluations, and use these as labels and dense move-level rewards.

### 2️⃣ RLVR (Reinforcement Learning with Verifiable Rewards)

This track uses Stockfish as a verifier and reward source.

Participants can:

* Implement a verifier that checks parsing and legality, and measures engine-based signals such as evaluation change, top-K alignment, and mate distance.
* Define scalar rewards from these signals, for example combining legality with binary or dense evaluation improvements and top-K matches.
* Use RL algorithms such as PPO or GRPO via Hugging Face TRL to optimise the policy with these verifiable rewards.

In both tracks, Stockfish serves as an objective reference for move quality, and the provided environment supplies the chess harness and metrics.

## 📥 Submission Format

Participants submit a chess agent that interacts with the provided environment through text. On each move, the environment sends a single chess position; the agent must reply with one move and one sentence.

### 🧩 Model input

For every turn, the agent receives a prompt that includes:

* **Position** encoded as a FEN string
* **Side to move** (White or Black)
* **List of legal moves** in UCI format (for example: `e2e4`, `g1f3`, `e7e8q`)

The agent does not need to compute legal moves or board state itself. All board logic is handled by the environment.

### 🧩 Model output

For each input position, the agent must return:

* Exactly  **one move** , chosen from the provided legal moves, in UCI format
* A **one-sentence rationale** in plain language explaining the idea behind the move

In the reference template, these are wrapped in simple tags:

* `<uci_move>...</uci_move>` for the move
* `<rationale>...</rationale>` for the explanation

Submissions are checked for:

* A single, correctly formatted UCI move
* The move being legal in the given position
* Presence of a short rationale

## 📊 Data & Environment

The challenge is built on open chess data and a shared tournament environment. Participants do not need to implement board logic or their own engine. Instead, they work with prepared text datasets and a harness that handles chess mechanics end to end.

### 📚 Data

The challenge data is derived from large, public chess corpora:

* **Game data**
* Based on the Lichess Open Database (CC0).
* Positions are extracted from different phases of the game (opening, middlegame, endgame).
* For each sampled position, the organisers construct tuples such as:
  `{FEN, side_to_move, legal_moves_uci, move_played, optional Stockfish labels}`.
* **Engine annotations**
* For many positions, Stockfish is run at a fixed depth or time to record:
  * best moves
  * principal variation (PV)
  * evaluation (centipawns or win/draw/loss scores)

A reference training set and example formats are provided in the starter kit. Participants can use these, extend them with additional open data, or bring their own compatible datasets.

All formats are text-based:

* **FEN** for positions
* **UCI** for moves
* **PGN** for storing full games played by agents during evaluation

### 🧱 Environment

All submissions run inside a common Python environment so that agents can be compared fairly.

The provided harness:

* uses `python-chess` for board representation, FEN/PGN handling, and legality checks
* integrates a local Stockfish engine for evaluation and baseline opponents
* supports multiple built-in agents (Random, First-move, Last-move, Stockfish at different skills, OpenAI / HF examples)
* runs tournaments using **TrueSkill** scheduling and produces:
* match results
* per-agent ratings
* PGN logs for all games played

## 🤖 Baseline Model

The challenge includes a reference baseline model to illustrate the task and provide a starting point.

### 🧠 Base model

* A compact, instruction-tuned model in the 7B–8B class (for example, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, or Mistral-7B-Instruct-v0.3).
* Parameter-efficient finetuning with LoRA or QLoRA.

### 🔁 Input–output format

The baseline is trained on a concise prompt format.

**Inputs include:**

* `Position (FEN): {FEN}`
* `Legal moves: {legal_moves_uci}`
* `Side to move: {side}`

**Outputs:**

* A one-sentence rationale inside `<rationale>...</rationale>`
* Exactly one UCI move inside `<uci_move>...</uci_move>`

### 🎯 Training objective and targets

* Supervised finetuning with cross-entropy on the UCI move tokens, plus auxiliary tokens for the rationale, bounded to roughly one sentence.
* Target validation behaviour:
* 99% legality rate
* 98% parse success
* Improved ACPL compared to Random, First, and Last baseline agents in the tournament harness.

## 📈 Evaluation & Metrics

Submissions are evaluated by playing full games in a shared tournament environment and by checking each move against strict format and engine-based criteria. The goal is to reward agents that are both strong players and reliable, rule-following reasoners.

### 1️⃣ Game-level performance

Game strength is measured by running round-robin tournaments between participant agents, Stockfish baselines, and simple reference agents (Random / First / Last) inside the provided `run_game.py` harness.

Key metrics:

* **TrueSkill rating**
* Each agent receives a TrueSkill rating with mean μ and uncertainty σ.
* The main score used for ranking is the **conservative rating** μ − 3σ.
* **Win / draw / loss vs baselines**
* Match results against Stockfish at different skill levels (for example, skill 1–5) at fixed time or depth.
* Results against simple agents (Random / First / Last) to ensure basic competence.

### 2️⃣ Move-level reliability

Agents must produce well-formed, legal, and sensible moves. For each position, the verifier checks the model’s output and compares it with Stockfish.

Move-level metrics include:

* **Parse success**
* Percentage of responses that contain exactly one correctly formatted UCI move in the expected tags.
* **Legality rate**
* Percentage of moves that are legal for the given position and side to move.
* **Engine alignment**
* How often the chosen move matches one of Stockfish’s top-K moves at a fixed depth.
* **Evaluation change / ACPL**
* Average change in engine evaluation after the model’s move (Δ centipawns or WDL shift).
* Average Centipawn Loss (ACPL) over a game, relative to Stockfish best play.

### 3️⃣ Rationale checks

The challenge also tracks basic properties of the textual explanations:

* Presence of a **single, concise rationale** for each move
* Optional heuristic checks on explanations (for example, whether common motifs such as “fork”, “back rank”, “passed pawn” align with the actual position)

## 🔑 Starter Kit

Make your first submission easily [using the starter k](https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit/)it. It includes a ready-to-use chess environment, baseline agents, and an example submission so participants can plug in a model and run games end to end with minimal setup.

**1. Environment up**

* Clone the chess environment repository.
* Run Stockfish versus a Random agent using `run_game.py`.
* Confirm PGN export and tournament JSON outputs are working.

**2. Data snack**

* Download one or two monthly Lichess dumps.
* Extract around one million positions.
* Cache Stockfish top-K moves and evaluations at a fixed depth or time budget.

**3. Supervised finetuning pass**

* Run QLoRA on the position-to-UCI task using the defined prompt format.
* Verify legality and parse success above 98–99%.
* Run a small tournament against Stockfish skill 1–3.

**4. RLVR pass (optional)**

* Plug verifier rewards into TRL PPO or GRPO.
* Run short rollouts and track evaluation change, ACPL, and TrueSkill (μ − 3σ).
* Iterate on reward thresholds and shaping.

## ⚙️ AWS Trainium

This challenge runs on [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/), with tooling provided through the [Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html) and [Hugging Face Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index).

Trainium are part of the purpose-built AI chips by AWS, enabling developers to achieve higher performance and lower costs, while building and deploying models. They are powered by the AWS Neuron is the developer stack for running deep learning and generative AI workloads on AWS Trainium and AWS Inferentia. Built on an open source foundation, Neuron enables developers to build, deploy and explore natively with PyTorch and JAX frameworks and with ML libraries such as HuggingFace, vLLM, PyTorch Lightning, and others.

Hugging Faces's Optimum Neuron library builds on Neuron and exposes familiar Trainer and Accelerator APIs and supports LoRA-style fine-tuning of 7B–8B models on Trainium chips.

Participants can refer to the provided resources to familiarise themselves with the setup:

* An [introductory video series](https://www.youtube.com/watch?v=9ihlYCzEuLQ&list=PLhr1KZpdzukedJbZLRqftcd9Dr7sXxKLm) on Trainium basics and fine-tuning workflows.
* A hands-on tutorial demonstrating supervised finetuning of an 8B Llama model on Trainium using LoRA or QLoRA adapters. (to be updated)

Trainium instances can be scaled across multiple chips using standard distributed training tools such as `torchrun` and `neuronx_distributed`, and checkpoints are stored in standard Hugging Face formats for portability.

## 🏆 Prizes

Cashprize pool: **USD 17,000** and Compute Credits of **USD 8000**

* 🥇 First Place: **USD 10,000** + USD 5000 credits
* 🥈 Second Place: **USD 5,000** + USD 2000 credits
* 🥉 Third Place: **USD 2,000** + USD 1,000 credits

## 📅 Timeline

* Challenge Launch: 2nd December, 2025 23:55 UCT
* Round 1 Ends: 31st December, 2025 23:55 UTC

  Further details on upcoming rounds will be updated here.
