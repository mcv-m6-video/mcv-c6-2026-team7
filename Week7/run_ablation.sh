#!/bin/bash

GPU=${1:-0}

CONFIGS=(
    ablation_baseline
    ablation_dropout
    ablation_smoothing
    ablation_dilated
    ablation_se
)

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: $cfg"
    echo "=========================================="
    python main_spotting.py --model "$cfg" --gpu "$GPU"
done

echo "All ablation runs completed."
