#!/bin/bash
#SBATCH -p H100,A100,RTX6000PRO
#SBATCH --exclude=node53
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=reflow_training_model_update
#SBATCH --output=/home/infres/yrothlin-24/HandFlow/logs/reflow/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/HandFlow/logs/reflow/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/HandFlow

mkdir -p logs/reflow checkpoints/reflow

python reflow_train.py \
    --parsing True \
    --model_config ../models/mdit.yaml
