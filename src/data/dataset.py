import os
import pickle
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, as_completed

from .splits import DataSplitter


class PyGGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_df: pd.DataFrame,
        split_name: str = 'train',
        max_nodes: int = 600,
        graph_suffix: str = ".pkl"
    ):
        """
        Initialize dataset from directory of individual pickled PyG graphs.
        
        Args:
            data_dir: Directory containing {protein_id}_{drug_id}.pkl files
            data_df: DataFrame with protein/drug/y columns for metadata
            split_name: Which split to use ('train', 'val', 'test', 'test_wt', 'test_mutation')
            max_nodes: Maximum nodes per graph (filters out larger graphs)
            graph_suffix: File extension for graph files
        """
        self.data_dir = Path(data_dir)
        self.data_df = data_df
        self.split_name = split_name
        self.max_nodes = max_nodes
        # number of threads for loading graphs
        self.num_workers = max(8, int((os.cpu_count() or 12) / 2))  # Limit to reasonable number of threads
        
        # Load all graph files
        graph_files = sorted(list(self.data_dir.glob(f"*{graph_suffix}")))
        # print(graph_files)

        # If a split DataFrame is provided, filter graph files by it
        if len(data_df) > 0:
            # Build allowed stems: protein_drug
            allowed = set((data_df['protein'].astype(str) + "_" + data_df['drug'].astype(str)).tolist())
            graph_files = [fp for fp in graph_files if fp.stem in allowed]

        logging.info(f"Loading {len(graph_files)} graphs from {self.data_dir} for split '{split_name}' using {self.num_workers} workers")

        # Load graphs from pickle files and filter them by max_nodes
        self.graphs = self._load_graphs(graph_files)
        logging.info(f"Loaded {len(graph_files)} graphs for split '{split_name}' from {self.data_dir}")


    def _load_single_graph(self, filename: Path, idx: int) -> Tuple[int, Optional[Data]]:
        """
        Load a single graph from pickle file.
        
        Args:
            filename: Path to the pickle file
            idx: Original index to preserve ordering
            
        Returns:
            Tuple of (index, graph) or (index, None) if loading fails
        """
        # logging.basicConfig(level=logging.INFO)
        # logging.info(f"Loading graph {filename}...")
        try:
            with open(filename, 'rb') as f:
                graph: Data = pickle.load(f)
            
            # These are already done by the new preprocessing procedure in src/preprocess/generate_datasets.py
            # # Validate max_nodes constraint
            # if graph.x.size(0) > self.max_nodes:
            #     logging.warning(
            #         f"Graph {filename} has {graph.x.size(0)} nodes > max_nodes {self.max_nodes}, skipping"
            #     )
            #     return (idx, None)
            
            # # Set graph properties
            # # graph.idx = idx
            # graph.y = graph.y.reshape(-1)
            # graph = preprocess_item(graph)
            
            return (idx, graph)
        
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            return (idx, None)


    def _load_graphs(self, graph_files: List[Path]) -> List[Data]:
        """
        Load graphs in parallel using multithreading while preserving order.
        
        Args:
            graph_files: List of paths to graph pickle files
            
        Returns:
            List of loaded graphs in the same order as graph_files
        """
        if self.max_nodes <= 0 or len(graph_files) == 0:
            return []

        # Dictionary to store results with their original indices
        results = {}

        # Use ThreadPoolExecutor because PyG data cannot be shared across processes
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks with their indices
            pool_futures = [
                executor.submit(self._load_single_graph, filename, idx)
                for idx, filename in enumerate(graph_files)
            ]

            # Collect results as they complete
            try:
                for i, future in enumerate(as_completed(pool_futures)):
                    idx, graph = future.result()
                    # skip None results
                    if graph is not None:
                        results[idx] = graph
                    if (i + 1) % 500 == 0 or (i + 1) == len(graph_files):
                        logging.info(f"Loaded {i + 1}/{len(graph_files)} graphs on split '{self.split_name}'")
            except KeyboardInterrupt:
                logging.error("KeyboardInterrupt received, terminating loading.")
                # let all workers terminate
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        # Reconstruct ordered list from results dictionary
        graphs = []
        for idx in sorted(results.keys()):
            graphs.append(results[idx])
        
        # Log statistics
        failed_count = len(graph_files) - len(graphs)
        if failed_count > 0:
            logging.warning(
                f"Failed to load {failed_count}/{len(graph_files)} graphs "
                f"(exceeded max_nodes or error)"
            )
        
        return graphs

    
    def __len__(self) -> int:
        return len(self.graphs)


    def __getitem__(self, idx: int) -> Data:
        """Load and return PyG graph."""
        return self.graphs[idx]


class MultiSplitDataset:
    """
    Wrapper class that manages multiple dataset splits.
    Matches the interface from dataset_GIGN_benchmark_davis_complete.py
    """
    def __init__(
        self,
        processed_dir: str,
        data_df_path: str,
        mmseqs_seq_clus_df_path: Optional[str]=None,
        split_method: str='random',
        max_nodes: int=512,
        seed: int=42,
        split_frac: Tuple[float, float, float]=(0.7, 0.1, 0.2)
    ):
        """
        Initialize multi-split dataset matching bingsong_project interface.
        
        Args:
            data_dir: Directory containing pickled PyG graph files
            data_df: DataFrame with protein/drug/y columns
            split_method: One of 9 splitting methods
            mmseqs_seq_clus_df: Required for sequence identity methods
            max_nodes: Maximum nodes per graph
            seed: Random seed
            split_frac: Train/val/test fractions
        """
        self.processed_dir = processed_dir

        self.data_df = pd.read_csv(data_df_path, sep='\t')
        # self.data_df['graph_filename'] = self.data_df['protein'].astype(str) + "_" + self.data_df['drug'].astype(str) + '.pkl'
        if mmseqs_seq_clus_df_path is not None:
            self.mmseqs_seq_clus_df = pd.read_table(mmseqs_seq_clus_df_path, names=['rep', 'seq'])
        else:
            self.mmseqs_seq_clus_df = None

        splitter = DataSplitter(self.data_df, self.mmseqs_seq_clus_df, seed=seed)
        self.split_indices_dict = splitter.generate_split_indices(split_method, split_frac)
        logging.info(f"Data splits: " + ", ".join([f"{k}: {len(v)}" for k, v in self.split_indices_dict.items()]))
        
        # Create datasets for each split
        self.datasets = {}
        for split_name, split_indices in self.split_indices_dict.items():
            if len(split_indices) > 0:  # Only create non-empty datasets
                self.datasets[split_name] = PyGGraphDataset(
                    data_dir=processed_dir,
                    data_df=self.data_df.iloc[split_indices].reset_index(drop=True),
                    split_name=split_name,
                    max_nodes=max_nodes,
                    graph_suffix=".pkl"
                )


    def get_dataset(self, split_name: str) -> PyGGraphDataset:
        """Get dataset for specific split."""
        if split_name not in self.datasets:
            raise ValueError(f"Split '{split_name}' not available. Available splits: {list(self.datasets.keys())}")
        return self.datasets[split_name]


    def get_split_sizes(self) -> Dict[str, int]:
        """Get sizes of all splits."""
        return {split: len(dataset) for split, dataset in self.datasets.items()}
