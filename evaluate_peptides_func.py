#!/usr/bin/env python3
"""
Evaluation script for Peptides-func dataset with and without attention improvement loss.
"""

import os
import sys
import argparse
import torch
import logging
from types import SimpleNamespace
from torch_geometric.graphgym.config import cfg, set_cfg, CN
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import register_config

# Import custom modules
from graphgps.finetuning import load_pretrained_model_cfg
from graphgps.train.custom_train import eval_epoch
from graphgps.logger import create_logger

# Import GraphGPS custom configs to register them
import graphgps.config


# Register attention loss config options
@register_config('attention_loss_cfg')
def attention_loss_cfg(cfg):
    """Config options for attention improvement loss."""
    cfg.model.use_attention_loss = False
    cfg.model.attention_loss_weight = 0.1
    cfg.model.attention_loss_tau = 0.2


def reset_cfg():
    """Reset the global cfg to default state."""
    # Clear all keys and reset to defaults
    cfg.defrost()
    for key in list(cfg.keys()):
        del cfg[key]
    set_cfg(cfg)


def evaluate_model(config_path, use_attention_loss=False, checkpoint_path=None):
    """
    Evaluate a model on Peptides-func dataset.
    
    Args:
        config_path: Path to config file
        use_attention_loss: Whether to use attention improvement loss
        checkpoint_path: Path to checkpoint (optional)
    """
    # Reset and reload config to ensure clean state
    reset_cfg()
    cfg.merge_from_file(config_path)
    
    # Override accelerator to use cuda:0 (or cpu)
    cfg.accelerator = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Override attention loss setting
    if not hasattr(cfg, 'model'):
        cfg.model = SimpleNamespace()
    cfg.model.use_attention_loss = use_attention_loss
    
    # Set a run directory for logging (required by graphgym logger)
    attention_str = "with_attn_loss" if use_attention_loss else "no_attn_loss"
    cfg.run_dir = os.path.join(os.path.dirname(config_path), f'eval_results/{attention_str}')
    os.makedirs(cfg.run_dir, exist_ok=True)
    
    # Set device - auto_select_device may not set cfg.device in newer versions
    auto_select_device()
    if not hasattr(cfg, 'device') or cfg.device is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(cfg.device)
    
    # Create data loaders
    loaders = create_loader()
    
    # Create logger
    loggers = create_logger()
    
    # Create model
    model = create_model()
    
    # Set cfg.params (required by eval_epoch) - count model parameters
    cfg.params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Ckpt path = ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    
    model.to(device)
    
    # Evaluate on test set
    logging.info(f"Evaluating with attention_loss={use_attention_loss}")
    eval_epoch(loggers[0], loaders[2], model, split='test')
    
    # Get results by calling write_epoch which computes and logs the metrics
    # For multilabel classification, we get accuracy, auc, and ap
    if cfg.dataset.task_type == 'classification_multilabel':
        test_results = loggers[0].classification_multilabel()
    elif cfg.dataset.task_type == 'classification_binary':
        test_results = loggers[0].classification_binary()
    elif cfg.dataset.task_type == 'classification_multi':
        test_results = loggers[0].classification_multi()
    elif cfg.dataset.task_type == 'regression':
        test_results = loggers[0].regression()
    else:
        test_results = loggers[0].basic()
    
    logging.info(f"Test results: {test_results}")
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphGPS on Peptides-func')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file for Peptides-func')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    print(args.__dict__)
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Evaluate without attention loss
    logging.info("=" * 80)
    logging.info("Evaluating WITHOUT attention improvement loss")
    logging.info("=" * 80)
    results_without = evaluate_model(args.config, use_attention_loss=False, 
                                     checkpoint_path=args.checkpoint)
    
    # Evaluate with attention loss
    logging.info("\n" + "=" * 80)
    logging.info("Evaluating WITH attention improvement loss")
    logging.info("=" * 80)
    results_with = evaluate_model(args.config, use_attention_loss=True,
                                  checkpoint_path=args.checkpoint)
    
    # Compare results
    logging.info("\n" + "=" * 80)
    logging.info("COMPARISON")
    logging.info("=" * 80)
    logging.info(f"Without attention loss: {results_without}")
    logging.info(f"With attention loss: {results_with}")


if __name__ == '__main__':
    main()
