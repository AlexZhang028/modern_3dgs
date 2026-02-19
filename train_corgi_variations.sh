#!/bin/bash

# Training Script for Corgi Parameter Tuning
# Usage: ./train_corgi_variations.sh

echo "Starting Training Variations - Corgi Dataset"
echo "--------------------------------------------"

# 1. High Temporal Learning Rate
echo "[1/3] Training High Temp LR..."
python train.py --config config/corgi_hightemp.yaml > corgi_hightemp.log 2>&1
echo "Done."

# 2. Dense Configuration
echo "[2/3] Training Dense Config..."
python train.py --config config/corgi_dense.yaml > corgi_dense.log 2>&1
echo "Done."

# 3. Lower Regularization
echo "[3/3] Training Low Reg Config..."
python train.py --config config/corgi_reg01.yaml > corgi_reg01.log 2>&1
echo "Done."

echo "All trainings complete."
