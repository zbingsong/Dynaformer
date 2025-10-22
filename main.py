import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.models import GraphormerModel
from src.data.dataloader import create_dataloaders
from src.train.train import Trainer, TrainerConfig, AMPConfig, FlagConfig
from src.eval.eval import Evaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Create and initialize model"""
    model = GraphormerModel.from_args(**model_config)
    return model


def create_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer"""
    optimizer_name = optimizer_config.get('optimizer', 'adam').lower()
    lr = optimizer_config.get('lr', 1e-4)
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    logging.info(f"Creating optimizer: {optimizer_name} with lr={lr}, weight_decay={weight_decay}")
    # print(type(lr), type(weight_decay))
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, scheduler_config: Dict[str, Any]) -> Optional[optim.lr_scheduler.LRScheduler]:
    """Create learning rate scheduler"""
    scheduler_name = scheduler_config.get('type', None)
    if scheduler_name is None:
        return None
    
    warmup_updates = scheduler_config.get('warmup_updates', 0)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_updates) if warmup_updates > 0 else None
    if scheduler_name == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == 'polynomial_decay':
        total_iters = scheduler_config.get('epochs', 100) - warmup_updates
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=scheduler_config.get('power', 1.0), last_epoch=-1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler] if warmup_scheduler else [scheduler],
        milestones=[warmup_updates] if warmup_scheduler else []
    )
    return scheduler


def train_mode(
        model: nn.Module, 
        dataloader_dict: Dict[str, DataLoader],
        config: Dict[str, Any], 
        checkpoint_dir: Optional[Path]=None,
        timestamp: str="000000_000000",
        device: str="cuda"
) -> None:
    """Training mode entry point"""
    logging.info("Starting training mode...")

    training_config = config['training']
    # Create optimizer
    optimizer = create_optimizer(model, training_config['optimizer'])
    scheduler = create_scheduler(optimizer, training_config['scheduler'])
    logging.info(f"Optimizer: {training_config['optimizer'].get('type', 'adam')}")
    if scheduler:
        logging.info(f"Scheduler: {training_config['scheduler'].get('type', 'polynomial_decay')}")
        logging.info(f"Warmup updates: {training_config['scheduler'].get('warmup_updates', 0)}")
    else:
        logging.info("No scheduler used")

    # Load checkpoint if provided
    if checkpoint_dir is not None and checkpoint_dir.exists():
        # Find latest checkpoint (format: checkpoint_epoch_{epoch}.pt)
        checkpoint_list = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoint_list:
            checkpoint_path = max(checkpoint_list, key=lambda p: int(p.stem.split("_")[-1]))
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 1) + 1
            existing_train_losses = checkpoint.get('train_losses', [])
            existing_val_losses = checkpoint.get('val_losses', [])
            logging.info(f"Resuming training from epoch {start_epoch}")
            del checkpoint
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")
    else:
        start_epoch = 1
        # Create training config
        checkpoint_dir = Path(f"{training_config['checkpoint_dir']}/{timestamp}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        existing_train_losses = None
        existing_val_losses = None
        logging.info(f"Created checkpoint directory at {checkpoint_dir}")
    
    trainer_config = TrainerConfig(
        flag_config=FlagConfig(**training_config['flag']),
        amp_config=AMPConfig(**training_config['amp']),
        clip_norm=training_config.get('clip_norm', 5.0),
        save_interval=training_config.get('save_interval', 5),
        num_epochs=training_config.get('epochs', 1),
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=training_config['criterion'],
        config=trainer_config,
        optimizer=optimizer,
        scheduler=scheduler,
        existing_train_losses=existing_train_losses,
        existing_val_losses=existing_val_losses,
        device=device
    )
    logging.info("Trainer initialized")

    # Save a copy of the config
    with open(checkpoint_dir / 'configs.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Start training
    logging.info("Starting training loop...")
    trainer.train(dataloader_dict['train'], dataloader_dict['valid'])

    # Remove all checkpoints
    # for checkpoint in checkpoint_dir.glob("*.pt"):
    #     checkpoint.unlink()
    # Save final model
    checkpoint_path = checkpoint_dir / 'model.pt'
    torch.save(model.state_dict(), checkpoint_path)
    
    logging.info(f"Model saved to {checkpoint_path}")


def eval_mode(
        model: nn.Module,
        dataloader_dict: Dict[str, DataLoader],
        checkpoint_dir: Path,
        device: str="cpu"
) -> None:
    """Evaluation mode entry point"""
    logging.info("Starting evaluation mode...")
    
    # Load checkpoint
    checkpoint_path = checkpoint_dir / 'model.pt'

    if checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Checkpoint loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device
    )
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_statistics = evaluator.evaluate(dataloader_dict['test'])
    test_wt_statistics = evaluator.evaluate(dataloader_dict.get('test_wt', dataloader_dict['test']))
    test_mutation_statistics = evaluator.evaluate(dataloader_dict.get('test_mutation', dataloader_dict['test']))
    # logging.info(f"Test Loss: {test_statistics:.4f}")
    
    # Save results
    results_path = checkpoint_dir / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write("Test Loss:\n")
        for key, value in test_statistics.items():
            f.write(f"{key}: {value:<.6f}\n")
        f.write("\nTest WT Statistics:\n")
        for key, value in test_wt_statistics.items():
            f.write(f"{key}: {value:<.6f}\n")
        f.write("\nTest Mutation Statistics:\n")
        for key, value in test_mutation_statistics.items():
            f.write(f"{key}: {value:<.6f}\n")
        f.write("\nTest evaluation completed successfully.\n")
    logging.info(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Dynaformer Training and Evaluation')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'eval', 'preprocess1', 'preprocess2', 'preprocess3'],
        default='train',
        help='Mode to run: train, eval, preprocess1, preprocess2, preprocess3'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint directory for continued training or evaluation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
    )
    # parser.add_argument(
    #     '--split',
    #     type=str,
    #     default='random',
    #     help='Data split method for preprocessing'
    # )
    
    args = parser.parse_args()
    if args.mode == 'eval' and args.checkpoint is None:
        parser.error("--checkpoint is required in eval mode")

    # Load configuration
    config = load_config(args.config if args.config is not None else os.path.join(args.checkpoint, 'configs.yaml'))
    # print(type(config['training']['lr']))
    # Setup logging
    logging.info(f"Configuration loaded from {args.config}")
    logging.info(f"Running in {args.mode} mode")

    # Set random seed for reproducibility
    seed = config['seed'] if 'seed' in config else args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')

    if args.mode == 'preprocess2':
        from src.preprocess.custom_input_individual import preprocess_main
        preprocess_main(
            tsv_path=Path(config['data']['data_df_path']),
            data_dir=Path(config['data'].get('raw_data_dir', './data/boltz')),
            output_dir=Path(config['data']['data_dir'])
        )
        return
    
    if args.mode == 'preprocess3':
        from src.preprocess import DataPreprocessor
        data_preprocessor = DataPreprocessor(
            processed_dir=config['data'].get('processed_dir', './data/processed'),
            data_dir=config['data'].get('data_dir', './data/pkl'),
            data_df_path=config['data']['data_df_path'],
            max_nodes=config['model'].get('max_nodes', 600),
        )
        data_preprocessor.generate_datasets()
        return    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set up device
    device = config['training']['device']
    # Create model
    logging.info("Initializing model...")
    model = create_model(config['model'])
    model = model.to(device)
    logging.info(f"Model created")

    # Create dataloaders
    split_method = config['data'].get('split_method', 'random')
    logging.info(f"Loading data with split method '{split_method}'...")
    dataloader_dict = create_dataloaders(
        processed_dir=config['data']['processed_dir'],
        data_df_path=config['data']['data_df_path'],
        mmseqs_seq_clus_df_path=config['data'].get('mmseqs_seq_clus_df_path', None),
        split_method=split_method,
        batch_size=config['training']['batch_size'],
        max_nodes=config['model']['max_nodes'],
        num_workers=config['data']['num_workers'],
        seed=seed,
        split_frac=config['data'].get('split_frac', (0.7, 0.1, 0.2)),
        mode=args.mode
    )
    # keys of dataloader_dict: 'train', 'valid', 'test', 'test_wt', 'test_mutation'
    if args.mode == 'train':
        config['seed'] = seed
        logging.info(f"Train samples: {len(dataloader_dict['train'].dataset)}, Val samples: {len(dataloader_dict['valid'].dataset)}")
    else:
        logging.info(f"Test samples: {len(dataloader_dict['test'].dataset)}, WT samples: {len(dataloader_dict.get('test_wt', dataloader_dict['test']).dataset)}, Mutation samples: {len(dataloader_dict.get('test_mutation', dataloader_dict['test']).dataset)}")

    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else None
    if args.mode == 'train':
        train_mode(model, dataloader_dict, config, checkpoint_dir, timestamp, device)
    elif args.mode == 'eval':
        assert checkpoint_dir is not None
        eval_mode(model, dataloader_dict, checkpoint_dir, device)

    logging.info(f"{args.mode.capitalize()} completed successfully")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main()
