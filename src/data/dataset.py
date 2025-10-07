import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
# from .preprocess import preprocess_item


class PyGGraphDataset(Dataset):
    """
    Modern PyTorch dataset for individual pickled PyG graphs.
    Supports all 9 splitting methods from bingsong_project.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        data_df: Optional[pd.DataFrame] = None,
        split_name: str = 'train',
        split_indices: Optional[List[int]] = None,
        max_nodes: int = 600,
        seed: int = 42,
        graph_suffix: str = ".pkl"
    ):
        """
        Initialize dataset from directory of individual pickled PyG graphs.
        
        Args:
            data_dir: Directory containing {protein_id}_{drug_id}.pkl files
            data_df: Optional DataFrame with protein/drug/y columns for metadata
            split_name: Which split to use ('train', 'val', 'test', 'test_wt', 'test_mutation')
            split_indices: Optional list of indices for this split
            max_nodes: Maximum nodes per graph (filters out larger graphs)
            seed: Random seed for reproducibility
            graph_suffix: File extension for graph files
        """
        self.data_dir = Path(data_dir)
        self.data_df = data_df
        self.split_name = split_name
        self.max_nodes = max_nodes
        self.seed = seed
        self.graph_suffix = graph_suffix
        
        # Load all graph files
        self.graph_files = sorted(list(self.data_dir.glob(f"*{graph_suffix}")))

        # If a split DataFrame is provided, filter graph files by it
        if data_df is not None and len(data_df) > 0:
            # Build allowed stems: protein_drug
            allowed = set(f"{str(row['protein']).strip()}_{str(row['drug']).strip()}" for _, row in data_df.iterrows())
            self.graph_files = [fp for fp in self.graph_files if fp.stem in allowed]
        
        # Apply split indices if provided
        if split_indices is not None:
            self.graph_files = [self.graph_files[i] for i in split_indices if i < len(self.graph_files)]
        
        # Parse metadata from filenames and DataFrame
        self._parse_metadata()
        
        # Filter graphs by max_nodes
        self._filter_by_size()
        
        print(f"Loaded {len(self.graph_files)} graphs for split '{split_name}' from {self.data_dir}")
    
    def _parse_metadata(self):
        """Extract protein_id, drug_id from filenames and merge with DataFrame info."""
        self.metadata = []
        
        # Create lookup from DataFrame if provided
        df_lookup = {}
        if self.data_df is not None:
            for _, row in self.data_df.iterrows():
                key = f"{row['protein']}_{row['drug']}"
                df_lookup[key] = {
                    'protein_id': row['protein'],
                    'drug_id': row['drug'],
                    'y': float(row['y']) if 'y' in row else 0.0
                }
        
        for file_path in self.graph_files:
            # Parse {protein_id}_{drug_id}.pkl format
            stem = file_path.stem
            
            # Try to match with DataFrame first
            if stem in df_lookup:
                meta = df_lookup[stem].copy()
                meta['filename'] = file_path.name
                self.metadata.append(meta)
            else:
                # Fall back to filename parsing
                # find index of last underscore
                last_underscore = stem.rfind('_')
                protein_id = stem[:last_underscore] if last_underscore != -1 else "unknown"
                drug_id = stem[last_underscore + 1:] if last_underscore != -1 else "unknown"

                self.metadata.append({
                    'protein_id': protein_id,
                    'drug_id': drug_id,
                    'filename': file_path.name,
                    'y': 0.0  # Default target
                })
    
    def _filter_by_size(self):
        """Filter out graphs exceeding max_nodes."""
        if self.max_nodes <= 0:
            return
        
        valid_indices = []
        for i, file_path in enumerate(self.graph_files):
            try:
                with open(file_path, 'rb') as f:
                    graph = pickle.load(f)
                if hasattr(graph, 'x') and graph.x.size(0) <= self.max_nodes:
                    valid_indices.append(i)
                else:
                    print(f"Filtered out {file_path.name}: {graph.x.size(0) if hasattr(graph, 'x') else 'N/A'} nodes > {self.max_nodes}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        self.graph_files = [self.graph_files[i] for i in valid_indices]
        self.metadata = [self.metadata[i] for i in valid_indices]
    
    def __len__(self) -> int:
        return len(self.graph_files)
    
    def __getitem__(self, idx: int) -> Data:
        """Load and return PyG graph."""
        file_path = self.graph_files[idx]
        
        try:
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
            
            # Ensure graph has required attributes
            if not hasattr(graph, 'x'):
                raise ValueError(f"Graph {file_path.name} missing node features 'x'")
            
            # Add target from metadata if not present
            if not hasattr(graph, 'y'):
                graph.y = torch.tensor([self.metadata[idx]['y']], dtype=torch.float)
            
            # Add metadata
            graph.protein_id = self.metadata[idx]['protein_id']
            graph.drug_id = self.metadata[idx]['drug_id']
            graph.split_name = self.split_name
            
            # graph = preprocess_item(graph)
            return graph
            
        except Exception as e:
            print(f"Error loading graph {file_path.name}: {e}")
            # Return dummy graph as fallback
            return self._create_dummy_graph()
    
    def _create_dummy_graph(self) -> Data:
        """Create dummy graph for error cases."""
        return Data(
            x=torch.zeros(1, 166),  # Standard node feature dim
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 15),
            pos=torch.zeros(1, 3),
            y=torch.tensor([0.0]),
            fp=torch.zeros(2040),  # RF+GB+ECIF features
            protein_id="dummy",
            drug_id="dummy",
            split_name=self.split_name
        )
    
    def get_split_info(self) -> Dict[str, List[str]]:
        """Get protein and drug IDs for split analysis."""
        proteins = [meta['protein_id'] for meta in self.metadata]
        drugs = [meta['drug_id'] for meta in self.metadata]
        return {'proteins': proteins, 'drugs': drugs}


class MultiSplitDataset:
    """
    Wrapper class that manages multiple dataset splits.
    Matches the interface from dataset_GIGN_benchmark_davis_complete.py
    """
    
    def __init__(
        self,
        data_dir: Path,
        data_df: pd.DataFrame,
        split_method: str,
        mmseqs_seq_clus_df: Optional[pd.DataFrame] = None,
        max_nodes: int = 600,
        seed: int = 42,
        split_frac: List[float] = [0.7, 0.1, 0.2]
    ):
        """
        Initialize multi-split dataset matching bingsong_project interface.
        
        Args:
            data_dir: Directory containing graph files
            data_df: DataFrame with protein/drug/y columns  
            split_method: One of 9 splitting methods
            mmseqs_seq_clus_df: Required for sequence identity methods
            max_nodes: Maximum nodes per graph
            seed: Random seed
            split_frac: Train/val/test fractions
        """
        self.data_dir = data_dir
        self.data_df = data_df
        self.split_method = split_method
        self.mmseqs_seq_clus_df = mmseqs_seq_clus_df
        self.max_nodes = max_nodes
        self.seed = seed
        self.split_frac = split_frac
        
        # Import and use DataSplitter
        from .splits import DataSplitter
        splitter = DataSplitter(seed=seed)
        
        # Create splits
        self.split_dfs = splitter.split_data(
            data_df, split_method, mmseqs_seq_clus_df, split_frac
        )
        
        # Create datasets for each split
        self.datasets = {}
        for split_name, split_df in self.split_dfs.items():
            if len(split_df) > 0:  # Only create non-empty datasets
                self.datasets[split_name] = PyGGraphDataset(
                    data_dir=data_dir,
                    data_df=split_df,
                    split_name=split_name,
                    max_nodes=max_nodes,
                    seed=seed
                )
    
    def get_dataset(self, split_name: str) -> PyGGraphDataset:
        """Get dataset for specific split."""
        if split_name not in self.datasets:
            raise ValueError(f"Split '{split_name}' not available. Available splits: {list(self.datasets.keys())}")
        return self.datasets[split_name]
    
    def get_split_sizes(self) -> Dict[str, int]:
        """Get sizes of all splits."""
        return {split: len(dataset) for split, dataset in self.datasets.items()}