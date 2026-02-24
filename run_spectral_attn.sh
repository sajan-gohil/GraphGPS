#!/bin/bash

# Script to evaluate the Spectral Attention model on Peptides-func dataset

# Activate your environment if needed
# source /path/to/your/venv/bin/activate

cd /home/srg/projects/GraphGPS

# Run the model with spectral attention
echo "Training Spectral Attention GPS model on Peptides-func..."
python main.py --config configs/GPS/peptides-func-GPS-SpectralAttention.yaml

# Optional: Run for more epochs with specific settings
# python main.py \
#   --config configs/GPS/peptides-func-GPS-SpectralAttention.yaml \
#   --opts train.max_epoch 1000 train.eval_period 10

# Optional: Run multiple seeds for statistical significance
# for seed in 0 1 2; do
#   echo "Running with seed $seed"
#   python main.py --config configs/GPS/peptides-func-GPS-SpectralAttention.yaml --seed $seed
# done
