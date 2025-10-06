import torch
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from pathlib import Path

from .dataset import PyGGraphDataset, MultiSplitDataset
from .splits import DataSplitter
from .collator import GraphormerCollator


def create_dataloaders_bingsong(
    data_dir: Path,
    data_df: pd.DataFrame,
    split_method: str,
    mmseqs_seq_clus_df: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    max_nodes: int = 600,
    num_workers: int = 4,
    seed: int = 42,
    split_frac: List[float] = [0.7, 0.1, 0.2]
) -> Dict[str, DataLoader]:
    """
    Create dataloaders using bingsong_project splitting methods.
    
    Args:
        data_dir: Directory containing individual pickled PyG graphs
        data_df: DataFrame with columns ['protein', 'drug', 'y']
        split_method: One of 9 methods from bingsong_project
        mmseqs_seq_clus_df: Required for sequence identity methods (columns: ['rep', 'seq'])
        batch_size: Batch size for dataloaders
        max_nodes: Maximum nodes per graph
        num_workers: Number of worker processes
        seed: Random seed
        split_frac: Train/val/test fractions
        
    Returns:
        Dictionary with available split dataloaders
    """
    # Initialize components
    collator = GraphormerCollator(max_nodes=max_nodes)
    
    # Create multi-split dataset
    multi_dataset = MultiSplitDataset(
        data_dir=data_dir,
        data_df=data_df,
        split_method=split_method,
        mmseqs_seq_clus_df=mmseqs_seq_clus_df,
        max_nodes=max_nodes,
        seed=seed,
        split_frac=split_frac
    )
    
    # Create dataloaders for each available split
    dataloaders = {}
    
    for split_name in ['train', 'valid', 'test', 'test_wt', 'test_mutation']:
        try:
            dataset = multi_dataset.get_dataset(split_name)
            if len(dataset) > 0:
                dataloaders[split_name] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split_name == 'train'),
                    collate_fn=collator,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
        except ValueError:
            # Split not available
            continue
    
    # Print split information
    split_sizes = multi_dataset.get_split_sizes()
    print(f"Split method: {split_method}")
    for split_name, size in split_sizes.items():
        print(f"  {split_name}: {size}")
    
    return dataloaders


def create_dataloaders(
    data_dir: Path,
    split_type: str = 'random',
    split_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    max_nodes: int = 600,
    num_workers: int = 4,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders with modern PyTorch implementation.
    Legacy interface maintained for backward compatibility.
    
    Args:
        data_dir: Directory containing individual pickled PyG graphs
        split_type: 'random', 'cold_drug', 'cold_protein', 'seq_identity', 'benchmark'
        split_params: Additional parameters for splitting
        batch_size: Batch size for dataloaders
        max_nodes: Maximum nodes per graph
        num_workers: Number of worker processes
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Initialize components
    splitter = DataSplitter(seed=seed)
    collator = GraphormerCollator(max_nodes=max_nodes)
    
    # Load full dataset to get metadata
    full_dataset = PyGGraphDataset(data_dir, max_nodes=max_nodes, seed=seed)
    metadata = full_dataset.metadata
    
    # Create splits
    if split_type == 'random':
        train_idx, test_idx = splitter.create_fold(
            len(full_dataset), 
            n_folds=split_params.get('n_folds', 5),
            fold_idx=split_params.get('fold_idx', 0)
        )
        # Use 10% of train for validation
        val_size = len(train_idx) // 10
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]
        
    elif split_type in ['cold_drug', 'cold_protein']:
        entity = split_type.split('_')[1]
        train_idx, test_idx = splitter.create_fold_setting_cold(
            metadata, entity_type=entity,
            test_ratio=split_params.get('test_ratio', 0.2)
        )
        val_size = len(train_idx) // 10
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]
        
    elif split_type == 'seq_identity':
        train_idx, test_idx = splitter.create_seq_identity_fold(
            metadata,
            identity_file=split_params.get('identity_file'),
            threshold=split_params.get('threshold', 0.5)
        )
        val_size = len(train_idx) // 10
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]
        
    elif split_type == 'benchmark':
        splits = splitter.create_benchmark_split(
            metadata,
            benchmark_file=split_params.get('benchmark_file')
        )
        train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
        
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    # Create datasets for each split
    train_dataset = PyGGraphDataset(data_dir, split_indices=train_idx, max_nodes=max_nodes)
    val_dataset = PyGGraphDataset(data_dir, split_indices=val_idx, max_nodes=max_nodes)
    test_dataset = PyGGraphDataset(data_dir, split_indices=test_idx, max_nodes=max_nodes)
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, 
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }
    
    print(f"Created dataloaders - Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return dataloaders