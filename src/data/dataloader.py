import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from pathlib import Path

from .dataset import PyGGraphDataset, MultiSplitDataset
from .splits import DataSplitter
from .collator import GraphormerCollator


def _build_df_from_dir(data_dir: Path) -> pd.DataFrame:
    """Build a minimal DataFrame with protein, drug, y by parsing filenames <protein>_<drug>.pkl."""
    rows = []
    for fp in sorted(Path(data_dir).glob("*.pkl")):
        stem = fp.stem
        if "_" in stem:
            parts = stem.split("_")
            protein = parts[0]
            drug = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
        else:
            protein, drug = stem, "unknown"
        # Try reading y from graph if present
        y_val = 0.0
        try:
            import pickle
            with open(fp, 'rb') as f:
                g = pickle.load(f)
            if hasattr(g, 'y') and g.y is not None:
                y_val = float(g.y[0]) if getattr(g.y, "numel", lambda: 0)() > 0 else float(g.y)
        except Exception:
            pass
        rows.append({"protein": protein, "drug": drug, "y": y_val})
    return pd.DataFrame(rows)


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
    
    for split_name in ['train', 'val', 'test', 'test_wt', 'test_mutation']:
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
    Backwards-compatible wrapper: build a DataFrame from data_dir and call create_dataloaders_bingsong.
    """
    split_params = split_params or {}
    # Map legacy split_type to bingsong split_method
    split_method_map = {
        'random': 'random',
        'cold_drug': 'drug_name',
        'cold_protein': 'protein_modification',
        'seq_identity': 'protein_seqid',
        'benchmark': 'random',  # fallback unless a benchmark file is provided
    }
    split_method = split_method_map.get(split_type, 'random')
    data_df = _build_df_from_dir(data_dir)
    mmseqs_df = split_params.get('identity_file')
    if isinstance(mmseqs_df, (str, Path)) and Path(mmseqs_df).exists():
        mmseqs_df = pd.read_csv(mmseqs_df, sep="\t", names=['rep','seq'])
    else:
        mmseqs_df = None
    return create_dataloaders_bingsong(
        data_dir=data_dir,
        data_df=data_df,
        split_method=split_method,
        mmseqs_seq_clus_df=mmseqs_df,
        batch_size=batch_size,
        max_nodes=max_nodes,
        num_workers=num_workers,
        seed=seed,
        split_frac=split_params.get('split_frac', [0.7, 0.1, 0.2])
    )