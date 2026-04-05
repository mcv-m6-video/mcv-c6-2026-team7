#!/bin/bash
#SBATCH --job-name=yolo_car_seg
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:quadro:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00

# 1. Load Cluster Modules
echo "=== Loading CUDA 12.2.2 ==="
module purge
module load CUDA/12.2.2

# 2. Load Conda Environment
echo "=== Activating Conda Environment ==="
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate yolo_seg_env

# 3. HPC Library Fix (CRITICAL for GLIBCXX errors)
# This tells the system to use the modern C++ libraries installed in your conda env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. Navigate to your project folder
# Make sure this path is correct for where your python script is located
cd /home/maguilar/master/C6/mcv-c6-2026-team7/Week5
# Run the automatic segmentation script

# Intermediate eval script also has the option to put as an arg --no_verbose to avoid printing
# the tables. Also, to change the number of epochs it does an eval it has to be changed in
# the script at the var EVAL_EVERY_PCT

for MODEL in baseline depthwise dino convnext tcn temporal_transformer ablation_backbone ablation_focal ablation_regularization ablation_backbone_focal distillation; do
    echo "=== Evaluating: $MODEL ==="
    python main_classification_intermediate_eval.py --model $MODEL --cfm
done

