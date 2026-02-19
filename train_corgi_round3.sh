#!/bin/bash

# Training Script for Corgi Parameter Tuning (Round 3)
# Based on Baseline 03, targeting motion blur elimination
# Usage: ./train_corgi_round3.sh

echo "Starting Training Variations (Round 3) - Corgi Dataset"
echo "--------------------------------------------"

# 1. Extended Densification
echo "[1/3] Training Extended Densification (50k iters, Long Densify)..."
python train.py --config config/corgi_ext_densify.yaml > corgi_ext_densify.log 2>&1
echo "Done."

# 2. Temporal Sharpness
echo "[2/3] Training Temporal Sharpness (High t_scale_lr)..."
python train.py --config config/corgi_t_sharp.yaml > corgi_t_sharp.log 2>&1
echo "Done."

# 3. Velocity Focus
echo "[3/3] Training Velocity Focus (High velocity_lr)..."
python train.py --config config/corgi_vel_focus.yaml > corgi_vel_focus.log 2>&1
echo "Done."

echo "All Round 3 trainings complete."
