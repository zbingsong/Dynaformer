import logging
import pickle
import os
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data import preprocess_item


class DataPreprocessor:
    def __init__(
        self,
        processed_dir: str,
        data_dir: str,
        data_df_path: str,
        max_nodes: int=600,
    ):
        self.processed_dir = Path(processed_dir)
        self.data_dir = Path(data_dir)
        self.data_df_path = Path(data_df_path)
        self.max_nodes = max_nodes

        self.num_workers = max(8, int((os.cpu_count() or 12) / 2))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.data_df = pd.read_csv(self.data_df_path, sep='\t')
        if len(self.data_df) == 0:
            raise ValueError(f"DataFrame loaded from {self.data_df_path} is empty!")
        
        self.data_df['graph_filename'] = self.data_df['protein'].astype(str) + "_" + self.data_df['drug'].astype(str) + '.pkl'


    def generate_datasets(self):
        logging.info(f"Generating datasets in {self.processed_dir}...")
        graph_filenames = self.data_df['graph_filename'].tolist()
        logging.info(f"    Processing {len(graph_filenames)} graphs")
        self._process_graphs(graph_filenames)
        logging.info("Dataset generation complete.")


    def _process_graphs(self, graph_filenames: List[str]) -> None:
        success_count = 0
        # use ProcessPoolExecutor for CPU-bound tasks
        # pytorch objects can't be shared between processes, so we need to save them to disk in _process_single_graph() and then reload
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks with their indices
            pool_futures = [
                executor.submit(self._process_single_graph, filename)
                for filename in graph_filenames
            ]

            # Collect results as they complete
            try:
                for i, future in enumerate(as_completed(pool_futures)):
                    result = future.result()
                    success_count += result
                    if (i + 1) % 100 == 0 or (i + 1) == len(graph_filenames):
                        logging.info(f"Processed {i + 1}/{len(graph_filenames)} graphs")
            except KeyboardInterrupt:
                logging.error("KeyboardInterrupt received, terminating loading.")
                # let all workers terminate
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        
        # Log statistics
        failed_count = len(graph_filenames) - success_count
        if failed_count > 0:
            logging.warning(
                f"Failed to process {failed_count}/{len(graph_filenames)} graphs "
                f"(exceeded max_nodes or error)"
            )


    def _process_single_graph(self, filename: str) -> bool:
        # check if it is already processed
        file_path = self.processed_dir / filename
        if file_path.exists():
            logging.info(f"Graph {filename} already processed, skipping.")
            return True
        
        file_path = self.data_dir / filename
        logging.info(f"Processing graph {filename}...")
        if not file_path.exists():
            logging.warning(f"File {filename} does not exist!")
            return False
        try:
            with open(file_path, 'rb') as f:
                graph: Data = pickle.load(f)
            
            # Validate max_nodes constraint
            if graph.x.size(0) > self.max_nodes:
                logging.warning(
                    f"Graph {filename} has {graph.x.size(0)} nodes > max_nodes {self.max_nodes}, skipping"
                )
                return False
            
            # Set graph properties
            graph.y = graph.y.reshape(-1)
            graph = preprocess_item(graph)
            with open(self.processed_dir / filename, 'wb') as f:
                pickle.dump(graph, f)
            return True
        
        except KeyboardInterrupt:
            logging.error("KeyboardInterrupt received, terminating loading.")
            raise

        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return False
