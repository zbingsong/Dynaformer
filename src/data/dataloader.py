import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple

from .dataset import PyGGraphDataset
from .collator import GraphormerCollator
from .splits import DataSplitter


def create_dataloaders(
    processed_dir: str,
    data_df_path: str,
    mmseqs_seq_clus_df_path: Optional[str]=None,
    split_method: str='random',
    batch_size: int=32,
    max_nodes: int=512,
    num_workers: int=4,
    seed: int=42,
    split_frac: Tuple[float, float, float]=(0.7, 0.1, 0.2),
    mode: str='train'
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

    data_df = pd.read_csv(data_df_path, sep='\t')
    if mmseqs_seq_clus_df_path is not None:
        mmseqs_seq_clus_df = pd.read_table(mmseqs_seq_clus_df_path, names=['rep', 'seq'])
    else:
        mmseqs_seq_clus_df = None

    splitter = DataSplitter(data_df, mmseqs_seq_clus_df, seed=seed)
    split_indices_dict = splitter.generate_split_indices(split_method, split_frac)
    logging.info(f"Data splits: " + ", ".join([f"{k}: {len(v)}" for k, v in split_indices_dict.items()]))
    
    if mode == 'train':
        # Only keep train/valid/test splits
        split_indices_dict = {k: v for k, v in split_indices_dict.items() if k in {'train', 'valid'}}
    else:
        # Only keep test/test_wt/test_mutation splits
        split_indices_dict = {k: v for k, v in split_indices_dict.items() if k in {'test', 'test_wt', 'test_mutation'}}

    # Create datasets for each split
    dataloaders: Dict[str, DataLoader] = {}
    for split_name, split_indices in split_indices_dict.items():
        if len(split_indices) > 0:  # Only create non-empty datasets
            dataset = PyGGraphDataset(
                data_dir=processed_dir,
                data_df=data_df.iloc[split_indices].reset_index(drop=True),
                split_name=split_name,
                max_nodes=max_nodes,
                graph_suffix=".pkl"
            )
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
    
    # Print split information
    logging.info(f"Split method: {split_method}")
    for split_name, dataloader in dataloaders.items():
        logging.info(f"  {split_name}: {len(dataloader.dataset)} samples")

    return dataloaders
