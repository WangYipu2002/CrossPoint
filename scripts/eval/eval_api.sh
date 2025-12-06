#!/bin/bash

# ===== Path Configuration =====
BENCHMARK_PATH="/your/downloaded/CrossPoint-Bench"  # Path to benchmark directory
RESULT_DIR="/your/workspace/eval_results/inference" # Directory to save results

# ===== API Configuration =====
API_KEY="your-api-key"    # Your API key
BASE_URL="your-base-url"  # API endpoint URL (e.g., https://api.openai.com/v1)

# ===== Model Configuration =====
# Add multiple models to evaluate, separated by spaces
MODELS=(
    gpt-4o-2024-11-20
    claude-3-7-sonnet
    gemini-2.5-pro
)

# ===== Optional: Number of parallel workers =====
MAX_WORKERS=8



for MODEL in "${MODELS[@]}"; do
    echo "====================================="
    echo "Evaluating model: ${MODEL}"
    echo "====================================="
    
    python eval/eval_api.py \
        --model "${MODEL}" \
        --api_key "${API_KEY}" \
        --base_url "${BASE_URL}" \
        --benchmark_path "${BENCHMARK_PATH}" \
        --result_dir "${RESULT_DIR}" \
        --max_workers ${MAX_WORKERS}
    
    echo ""
done

echo "All evaluations completed!"

