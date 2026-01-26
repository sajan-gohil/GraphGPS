# Attention Improvement Loss for GraphGPS

This implementation adds an attention improvement loss function to the transformer layers in GraphGPS to encourage better structural associations in the learned representations.

## Overview

The attention improvement loss encourages the model to improve similarity scores for connected nodes after the attention transformation. It computes a soft recall metric based on whether the final similarity scores exceed the mean initial similarity score for each graph.

## Implementation Files

1. **graphgps/loss/attention_improvement_loss.py**: Core loss function
2. **graphgps/layer/gps_layer.py**: Modified to compute and accumulate the loss
3. **graphgps/network/gps_model.py**: Modified to pass config parameters to layers
4. **graphgps/train/custom_train.py**: Modified to add the weighted loss during training
5. **evaluate_peptides_func.py**: Evaluation script to compare with/without the loss

## Configuration

Add the following parameters to your config file (e.g., `configs/GPS/peptides-func-GPS.yaml`):

```yaml
model:
  type: GPSModel
  loss_fun: cross_entropy
  graph_pooling: mean
  # Attention improvement loss parameters
  use_attention_loss: True          # Enable the attention improvement loss
  attention_loss_weight: 0.1        # Weight for combining with main loss
  attention_loss_tau: 0.2           # Temperature for sigmoid in loss computation
```

### Parameters:

- `use_attention_loss` (bool, default: False): Enable/disable the attention improvement loss
- `attention_loss_weight` (float, default: 0.1): Weight coefficient for the loss. The total loss is: `total_loss = task_loss + attention_loss_weight * attention_loss`
- `attention_loss_tau` (float, default: 0.2): Temperature parameter for the sigmoid function in the loss computation

## Usage

### Training with Attention Loss

```bash
# Using the example config with attention loss enabled
python main.py --cfg configs/GPS/peptides-func-GPS-attn-loss.yaml

# Or override in command line
python main.py --cfg configs/GPS/peptides-func-GPS.yaml \
    model.use_attention_loss True \
    model.attention_loss_weight 0.1 \
    model.attention_loss_tau 0.2
```

### Evaluation Script

Compare model performance with and without attention improvement loss:

```bash
python evaluate_peptides_func.py --config configs/GPS/peptides-func-GPS.yaml
```

With a checkpoint:

```bash
python evaluate_peptides_func.py \
    --config configs/GPS/peptides-func-GPS.yaml \
    --checkpoint results/peptides-func/best_model.pt
```

The script will:
1. Evaluate the model on the test set WITHOUT attention loss
2. Evaluate the model on the test set WITH attention loss
3. Display a comparison of the results

## How It Works

1. **Loss Computation**: For each GPS layer with self-attention:
   - Node embeddings are stored before the attention transformation
   - After attention (but before residual connection), the attention improvement loss is computed
   - The loss measures how well the attention improves similarity scores for connected nodes

2. **Loss Formula**:
   - Normalize embeddings before and after attention
   - Compute similarity scores for connected edges
   - Use per-graph mean initial score as threshold
   - Apply sigmoid to compute soft recall: `sigmoid((final_score - threshold) / tau)`
   - Loss is negative log-likelihood of recall, averaged across graphs

3. **Training**: The attention loss is accumulated across all GPS layers and added to the main task loss with a configurable weight.

## Backward Compatibility

The implementation is fully backward compatible. If you don't specify the attention loss parameters in your config:
- `use_attention_loss` defaults to `False`
- No additional computation is performed
- Models behave exactly as before

## Example Configs

Two example configs are provided for the Peptides-func dataset:

1. **peptides-func-GPS.yaml**: Original config (no attention loss)
2. **peptides-func-GPS-attn-loss.yaml**: Config with attention loss enabled
