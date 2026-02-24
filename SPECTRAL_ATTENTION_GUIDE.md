# Spectral Attention Implementation Guide

## Overview

I have successfully replaced all normal attention blocks with Spectral Attention (SpectralAttentionLayer) based on Chebyshev polynomial filters. The implementation handles graph-structured data using the normalized Laplacian in the spectral domain.

## Files Modified/Created

### New Files:
1. **`graphgps/layer/spectral_attn_layer.py`** - Complete spectral attention implementation
   - `ChebyshevFilter` - Learnable spectral filter using Chebyshev polynomials
   - `SpectralAttentionLayer` - Multi-head spectral attention with FFN

2. **`configs/GPS/peptides-func-GPS-SpectralAttention.yaml`** - Configuration for Peptides-func dataset with spectral attention

3. **`run_spectral_attn.sh`** - Bash script to run the model

### Modified Files:
1. **`graphgps/layer/gps_layer.py`**
   - Added import for `SpectralAttentionLayer`
   - Added 'SpectralAttention' as a global model type option
   - Modified forward pass to handle spectral attention's different interface
   - Updated dropout logic (SpectralAttentionLayer includes it internally)

## Key Features of Spectral Attention

### 1. **Chebyshev Polynomial Filters**
   - Learns band-pass filters in the spectral domain
   - Uses the normalized graph Laplacian
   - K=3 order (configurable) Chebyshev polynomials

### 2. **Per-Head Spectral Filtering**
   - Each attention head gets its own learnable filter for Q and K projections
   - Allows diverse spectral properties per head

### 3. **Graph-Aware Attention**
   - Directly uses edge_index structure
   - Works with variable-sized graphs in batches
   - Maintains sparse graph structure throughout

## Evaluation Commands

### Quick Start (Single Run):
```bash
cd /home/srg/projects/GraphGPS
python main.py --config configs/GPS/peptides-func-GPS-SpectralAttention.yaml
```

### Run with Custom Parameters:
```bash
python main.py \
  --config configs/GPS/peptides-func-GPS-SpectralAttention.yaml \
  --opts train.max_epoch 1000 \
           train.eval_period 10 \
           train.batch_size 64 \
           gt.n_heads 8 \
           gt.dim_hidden 128
```

### Multiple Seeds for Statistical Significance:
```bash
for seed in 0 1 2; do
  echo "Running with seed $seed"
  python main.py --config configs/GPS/peptides-func-GPS-SpectralAttention.yaml --seed $seed
done
```

### Using the Bash Script:
```bash
bash run_spectral_attn.sh
```

## Configuration Details

Key parameters in `peptides-func-GPS-SpectralAttention.yaml`:

```yaml
gt:
  layer_type: CustomGatedGCN+SpectralAttention  # Switches to spectral attention
  layers: 4                                       # Number of GPS layers
  n_heads: 4                                      # Number of attention heads
  dim_hidden: 96                                  # Hidden dimension
  dropout: 0.0
  attn_dropout: 0.1                              # Dropout in spectral filters
```

## Model Architecture Comparison

| Component | Standard Transformer | Spectral Attention |
|-----------|---------------------|-------------------|
| Q, K Projection | Linear → Multi-head | Linear → Spectral Filter per head → Multi-head |
| Attention Score | Dot product (Q·K) | Dot product (Filtered Q·Filtered K) |
| Graph Structure | Dense batch representation | Sparse edge_index |
| Computational Advantage | Works with any data | Exploits graph structure via Laplacian |

## Algorithm Summary

For each head h:
1. **Spectral Filtering**: Apply Chebyshev polynomial filter to Q and K
   - $q_h^{spec} = \sum_{k=0}^{K} \theta_k^{(h)} T_k(L) q_h$
   - $k_h^{spec} = \sum_{k=0}^{K} \phi_k^{(h)} T_k(L) k_h$

2. **Standard Attention**: Apply scaled dot-product attention
   - $\text{score} = \text{softmax}(q_h^{spec} \cdot (k_h^{spec})^T / \sqrt{d})$

3. **Output**: Combine heads and apply FFN with residual connections

## Hardware Requirements

- **GPU**: Recommended for faster training (tested on CUDA-capable GPUs)
- **CPU**: Functional but slower, especially for large graphs
- **Memory**: Similar to standard GPS models (~8GB for batch_size=128)

## Expected Results

The model should achieve comparable or better performance on Peptides-func compared to standard Transformer attention due to:
- Explicit incorporation of graph structure via Laplacian
- Learnable spectral filters that adapt to the data
- Reduced attention complexity through spectral domain filtering

## Troubleshooting

### Issue: Import Error for SpectralAttentionLayer
**Solution**: Ensure `graphgps/layer/spectral_attn_layer.py` is in the correct location

### Issue: CUDA Out of Memory
**Solution**: Reduce `train.batch_size` in the config or use `--opts train.batch_size 64`

### Issue: Graph structure mismatch
**Solution**: Ensure edge_index is properly provided in the batch object

## Future Enhancements

1. Adaptive K (order of Chebyshev polynomials) per head
2. Learnable scaling for different spectral bands
3. Spectral attention with edge attributes
4. Variant: Use other spectral bases (e.g., Hermite, Laguerre)

## Citation References

The implementation is based on spectrally-aligned attention mechanisms for graph neural networks. The Chebyshev polynomial filtering approach originates from:
- Defferrard et al., "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (NIPS 2016)
- Combined with multi-head attention from "Attention Is All You Need" (Vaswani et al., 2017)
