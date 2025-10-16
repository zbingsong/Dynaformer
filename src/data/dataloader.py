import logging
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple

from .dataset import MultiSplitDataset
from .collator import GraphormerCollator


def create_dataloaders(
    processed_dir: str,
    data_df_path: str,
    mmseqs_seq_clus_df_path: Optional[str]=None,
    split_method: str='random',
    batch_size: int=32,
    max_nodes: int=512,
    num_workers: int=4,
    seed: int=42,
    split_frac: Tuple[float, float, float]=(0.7, 0.1, 0.2)
) -> Dict[str, DataLoader]:
    """
    Create dataloaders using bingsong_project splitting methods.
    
    Args:
        data_dir: Directory containing individual pickled PyG graphs
        data_df_path: Path to the DataFrame with columns ['protein', 'drug', 'y']
        mmseqs_seq_clus_df_path: Required for sequence identity methods (columns: ['rep', 'seq'])
        split_method: One of 9 methods from bingsong_project
        batch_size: Batch size for dataloaders
        max_nodes: Maximum nodes per graph
        num_workers: Number of worker processes for data loading
        seed: Random seed
        split_frac: Train/val/test fractions
        
    Returns:
        Dictionary with available split dataloaders
    """
    # Initialize components
    collator = GraphormerCollator(max_nodes=max_nodes)
    
    # Create multi-split dataset
    multi_dataset = MultiSplitDataset(
        processed_dir=processed_dir,
        data_df_path=data_df_path,
        mmseqs_seq_clus_df_path=mmseqs_seq_clus_df_path,
        split_method=split_method,
        max_nodes=max_nodes,
        seed=seed,
        split_frac=split_frac
    )
    
    # Create dataloaders for each available split
    dataloaders = {}
    
    split_sizes = multi_dataset.get_split_sizes()
    for split_name, split_size in split_sizes.items():
        if split_size > 0:
            dataset = multi_dataset.get_dataset(split_name)
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
    
    # Print split information
    split_sizes = multi_dataset.get_split_sizes()
    logging.info(f"Split method: {split_method}")
    for split_name, size in split_sizes.items():
        logging.info(f"  {split_name}: {size}")

    return dataloaders
