#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
NUM_GPUS=2
CONFIG_FILE="accelerate_config.yaml"

# generate a temporary accelerate config to ensure 2-GPU bf16 mode works 
# without needing manual interactive setup
cat <<EOT > $CONFIG_FILE
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT

echo "🚀 Launching GRPO training on $NUM_GPUS GPUs..."

# Launch command
nohup accelerate launch \
    --config_file $CONFIG_FILE \
    train_grpo.py &> nohup.out &

# Cleanup (optional)
# rm $CONFIG_FILE