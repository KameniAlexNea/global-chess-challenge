# global-chess-challenge

Train LLMs to play chess (legal UCI move) and produce a short rationale.

This repo contains:
- SFT + GRPO training scripts
- Dataset processing utilities
- A local evaluation workflow using the official AIcrowd starter kit

## Setup (uv)

This project is managed with `uv` via [pyproject.toml](pyproject.toml).

```bash
uv sync
```

Run commands either by activating the venv:

```bash
source .venv/bin/activate
```

…or by prefixing with `uv run`:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## GRPO training (Accelerate)

The GRPO training entrypoint is [train_grpo.py](train_grpo.py). It is written to be launched with one process per GPU.

1 GPU:

```bash
uv run accelerate launch --num_processes 1 train_grpo.py
```

2 GPUs:

```bash
uv run accelerate launch --num_processes 2 train_grpo.py
```

Outputs are saved under `models/chess-grpo-sequences/` (adapter + merged model).

## Run the trained GRPO model

You can run either:
- the model you trained locally (e.g. `models/chess-grpo-sequences/`), or
- the published model on Hugging Face: `alexneakameni/Qwen2.5-Coder-0.5B-Instruct-chess-grpo`

### Option A: Quick local inference (Transformers)

This runs a single forward generation from a FEN, using the same prompt structure as training.

```bash
uv run python - <<'PY'
import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import system_msg, user_msg
from src.tokenizer_utils import ensure_chat_template

MODEL = "alexneakameni/Qwen2.5-Coder-0.5B-Instruct-chess-grpo"  # or "models/chess-grpo-sequences"
FEN = "r2q1rk1/ppp2pbp/2np1np1/4P3/4PB2/2N2B2/PPPQ1PPP/2KR3R b - - 0 3"

tokenizer = AutoTokenizer.from_pretrained(MODEL, fix_mistral_regex=True)
tokenizer = ensure_chat_template(tokenizer)
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
	MODEL,
	torch_dtype=torch.bfloat16,
	device_map={"": 0} if torch.cuda.is_available() else None,
)

board = chess.Board(FEN)
side_to_move = "White" if board.turn == chess.WHITE else "Black"
legal_moves_uci = " ".join(m.uci() for m in board.legal_moves)

prompt = user_msg.format(
	FEN=board.fen(),
	side_to_move=side_to_move,
	legal_moves_uci=legal_moves_uci,
)

messages = [
	{"role": "system", "content": system_msg},
	{"role": "user", "content": prompt},
]

chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat, return_tensors="pt").to(model.device)

with torch.no_grad():
	out = model.generate(
		**inputs,
		max_new_tokens=64,
		do_sample=True,
		temperature=1.0,
		top_p=0.95,
		top_k=64,
		pad_token_id=tokenizer.pad_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)

completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(completion)
PY
```

The completion is expected to contain `<rationale>...</rationale><uci_move>...</uci_move>`.

### Option B: Run the model as a vLLM server + local evaluation

This uses the official AIcrowd starter kit’s OpenAI-compatible evaluation client.

Terminal 1 (start vLLM):

```bash
cd global-chess-challenge-2025-starter-kit/player_agents
MODEL_NAME_OR_PATH="alexneakameni/Qwen2.5-Coder-0.5B-Instruct-chess-grpo" bash run_vllm.sh
```

Terminal 2 (run evaluation):

```bash
cd global-chess-challenge-2025-starter-kit
uv run python local_evaluation.py --endpoint http://localhost:5000/v1 \
  --template-file player_agents/qwen_prompt.jinja \
  --games-per-opponent 10
```

Notes:
- If you want to evaluate your *local* trained model, set `MODEL_NAME_OR_PATH="../../models/chess-grpo-sequences"` in the vLLM command.
- The evaluation script expects the move inside `<uci_move>...</uci_move>`.
