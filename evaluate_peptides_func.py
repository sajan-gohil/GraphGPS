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
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device

# Import custom modules
from graphgps.finetuning import load_pretrained_model_cfg
from graphgps.train.custom_train import eval_epoch
from graphgps.logger import create_logger


def evaluate_model(config_path, use_attention_loss=False, checkpoint_path=None):
    """
    Evaluate a model on Peptides-func dataset.
    
    Args:
        config_path: Path to config file
        use_attention_loss: Whether to use attention improvement loss
        checkpoint_path: Path to checkpoint (optional)
    """
    # Parse args and load config
    args = parse_args()
    args.cfg_file = config_path
    set_cfg(cfg)
    cfg.merge_from_file(config_path)
    
    # Override attention loss setting
    if not hasattr(cfg, 'model'):
        cfg.model = SimpleNamespace()
    cfg.model.use_attention_loss = use_attention_loss
    
    # Set device
    auto_select_device()
    
    # Create data loaders
    loaders = create_loader()
    
    # Create logger
    loggers = create_logger()
    
    # Create model
    model = create_model()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    
    model.to(cfg.device)
    
    # Evaluate on test set
    logging.info(f"Evaluating with attention_loss={use_attention_loss}")
    eval_epoch(loggers[0], loaders[2], model, split='test')
    
    # Get results
    test_results = loggers[0].get_stats()
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphGPS on Peptides-func')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file for Peptides-func')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
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
    set_printing()
    main()
