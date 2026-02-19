#!/bin/bash

# Training Script for Corgi Parameter Tuning (Round 2)
# Fix: lambda_reg set to 0.01 to avoid opacity crushing
# Usage: ./train_corgi_fix.sh

echo "Starting Training Variations (Round 2) - Corgi Dataset"
echo "--------------------------------------------"

# 1. Motion Fix
echo "[1/3] Training Motion Fix (High Temp/Vel LR, Safe Reg)..."
python train.py --config config/corgi_motion_fix.yaml > corgi_motion_fix.log 2>&1
echo "Done."

# 2. Dense Fix
echo "[2/3] Training Dense Fix (Dense Points, Safe Reg)..."
python train.py --config config/corgi_dense_fix.yaml > corgi_dense_fix.log 2>&1
echo "Done."

# 3. Sharpness Fix
echo "[3/3] Training Sharp Fix (High Scale LR, Safe Reg)..."
python train.py --config config/corgi_sharp_fix.yaml > corgi_sharp_fix.log 2>&1
echo "Done."

echo "All trainings complete."
