MODEL_NAME_OR_PATH="alexneakameni/Qwen2.5-Coder-0.5B-Instruct-chess-grpo"
MODEL_NAME_OR_PATH="models/chess-sft-conversation/merged-checkpoint-3000"

vllm serve $MODEL_NAME_OR_PATH \
    --served-model-name aicrowd-chess-model \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --disable-log-stats \
    --host 0.0.0.0 \
    --port 5000