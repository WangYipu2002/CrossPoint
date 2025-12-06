#!/bin/bash

# ===== Path Configuration =====
BENCHMARK_PATH="your/path/to/CrossPoint-Bench"  # Path to benchmark directory /your/downloaded/CrossPoint-Bench
RESULT_DIR="your/path/to/eval_results" # Directory to save results  /your/workspace/eval_results

# ===== Model Configuration =====
MODELS=(
    #"/your/model/checkpoint"
    #"your/path/to/Qwen2.5-VL-3B-Instruct"
    #"your/path/to/another/checkpoint"
)

# Loop through each model for evaluation
for MODEL_PATH in "${MODELS[@]}"; do

    python eval/eval_opensource.py \
        --model "${MODEL_PATH}" \
        --benchmark_path "${BENCHMARK_PATH}" \
        --result_dir "${RESULT_DIR}/inference"
    
    echo ""
done
