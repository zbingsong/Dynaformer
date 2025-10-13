import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from src.models import GraphormerModel
from src.data.dataloader import create_dataloaders
from src.train.train import Trainer, TrainingConfig
from src.eval.eval import Evaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_config: Dict[str, Any], device: str) -> nn.Module:
    """Create and initialize model"""
    model = GraphormerModel.from_args(**model_config)
    model = model.to(device)
    if model_config.get('fp16', False):
        model = model.half()
    return model


def create_optimizer(model: nn.Module, training_config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer"""
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    lr = training_config.get('lr', 1e-4)
    weight_decay = training_config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def train_mode(config: Dict[str, Any]):
    """Training mode entry point"""
    logging.info("Starting training mode...")
    
    # Setup device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Set random seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    logging.info("Initializing model...")
    model = create_model(config['model'], device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = create_optimizer(model, config['training'])
    logging.info(f"Optimizer: {config['training'].get('optimizer', 'adam')}")

    # Create dataloaders
    logging.info("Loading data...")
    dataloader_dict = create_dataloaders(
        data_dir=config['data']['data_dir'],
        data_df_path=config['data']['data_df_path'],
        mmseqs_seq_clus_df_path=config['data'].get('mmseqs_seq_clus_df_path', None),
        split_method=config['data']['split_method'],
        batch_size=config['training']['batch_size'],
        max_nodes=config['data']['max_nodes'],
        num_workers=config['data']['num_workers'],
        seed=seed,
        split_frac=config['data'].get('split_frac', (0.7, 0.1, 0.2)),
    )
    logging.info(f"Train samples: {len(dataloader_dict['train'].dataset)}, Val samples: {len(dataloader_dict['val'].dataset)}")
    
    # Create training config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = Path(f"{config['data']['checkpoint_dir']}_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    training_config = TrainingConfig(
        use_flag=config['flag'].get('flag', False),
        flag_m=config['flag'].get('flag_m', 3),
        flag_step_size=config['flag'].get('flag_step_size', 1e-3),
        flag_mag=config['flag'].get('flag_mag', 1e-3),
        use_amp=config['model'].get('fp16', False),
        log_interval=config['training'].get('log_interval', 10),
        num_epochs=config['training']['epochs'],
        checkpoint_dir=config['data']['checkpoint_dir'],
        results_dir=config['data']['results_dir'],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=config['training']['criterion'],
        optimizer=optimizer,
        config=training_config,
        device=device
    )
    
    # Start training
    logging.info("Starting training loop...")
    trainer.train(dataloader_dict['train'], dataloader_dict['val'])

    # Remove all checkpoints
    # for checkpoint in checkpoint_dir.glob("*.pt"):
    #     checkpoint.unlink()
    # Save final model
    checkpoint_path = checkpoint_dir / 'model.pt'
    torch.save(model.state_dict(), checkpoint_path)
    # Save a copy of the config
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    logging.info(f"Model saved to {checkpoint_path}")


def eval_mode(config: Dict[str, Any], checkpoint_dir: str):
    """Evaluation mode entry point"""
    logging.info("Starting evaluation mode...")
    
    # Setup device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Set random seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    logging.info("Initializing model...")
    model = create_model(config['model'], device)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_dir) / 'model.pt'

    if checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Checkpoint loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Create dataloaders
    logging.info("Loading data...")
    dataloader_dict = create_dataloaders(
        data_dir=config['data']['data_dir'],
        data_df_path=config['data']['data_df_path'],
        mmseqs_seq_clus_df_path=config['data'].get('mmseqs_seq_clus_df_path', None),
        split_method=config['data']['split_method'],
        batch_size=config['training']['batch_size'],
        max_nodes=config['data']['max_nodes'],
        num_workers=config['data']['num_workers'],
        seed=seed,
        split_frac=config['data'].get('split_frac', (0.7, 0.1, 0.2)),
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        criterion=config['training']['criterion'],
        device=device
    )
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_loss = evaluator.evaluate(dataloader_dict['test'])
    logging.info(f"Test Loss: {test_loss:.4f}")
    
    # Save results
    results_dir = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
    logging.info(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Dynaformer Training and Evaluation')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'eval'],
        help='Mode to run: train or eval'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logging.info(f"Configuration loaded from {args.config}")
    logging.info(f"Running in {args.mode} mode")
    
    if args.mode == 'train':
        train_mode(config)
    elif args.mode == 'eval':
        eval_mode(config, args.checkpoint)

    logging.info(f"{args.mode.capitalize()} completed successfully")


if __name__ == '__main__':
    main()