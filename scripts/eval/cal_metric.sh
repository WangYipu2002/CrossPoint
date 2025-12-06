#!/bin/bash

# ===== Configuration =====
RESULT_DIR="your/path/to/eval_results"
COORD_FORMAT="absolute" # absolute, relative_1000, relative_1 (depend on your model's output format)


# ===== Step 1: Extract Answers =====
shopt -s nullglob
for eval_file in "${RESULT_DIR}"/inference/eval_*.jsonl; do
    python eval/extract_answers.py \
        --file "$eval_file" \
        --coord_format "$COORD_FORMAT"
done
shopt -u nullglob

# ===== Step 2: Calculate Scores =====
python eval/calc_score.py \
    --extracted_root "${RESULT_DIR}/extracted" \
    --output_dir "${RESULT_DIR}/scores"


