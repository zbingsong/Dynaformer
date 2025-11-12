import logging
import pickle
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import RDLogger

from .preprocess import gen_feature, gen_graph, to_pyg_graph, get_info, RF_score, GB_score
from .ecif import GetECIF
from .utils import read_mol, obabel_pdb2mol, pymol_pocket
from .preprocess_graph import preprocess_item


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
        self.data_df["protein"] = self.data_df["protein"].astype(str).str.strip()
        self.data_df["drug"] = self.data_df["drug"].astype(str).str.strip()
        self.data_df["y"] = self.data_df["y"].astype(float)


    def generate_datasets(self):
        logging.info(f"Generating datasets in {self.processed_dir}...")
        receptors, ligands, names, pks = self._build_lists()
        logging.info(f"    Processing {len(receptors)} graphs")
        self._process_graphs(receptors, ligands, names, pks)
        logging.info("Dataset generation complete.")


    def _build_lists(self) -> tuple[list[str], list[str], list[str], list[float]]:
        receptors: list[str] = []
        ligands: list[str] = []
        names: list[str] = []
        pks: list[float] = []

        for idx, row in self.data_df.iterrows():
            protein = row["protein"]
            drug = row["drug"]
            y = float(row["y"])

            base_name = f"{protein}_{drug}"
            pdb_path = self.data_dir / f"{base_name}_model_0_protein.pdb"
            sdf_path = self.data_dir / f"{base_name}_model_0_ligand.sdf"

            if not pdb_path.exists() or not sdf_path.exists():
                msg = f"Missing files for row {idx} ({base_name}): " \
                    f"{'PDB missing' if not pdb_path.exists() else ''} " \
                    f"{'SDF missing' if not sdf_path.exists() else ''}"
                logging.warning(msg)
                continue  # Skip this entry if files are missing

            receptors.append(str(pdb_path.resolve()))
            ligands.append(str(sdf_path.resolve()))
            names.append(base_name)
            pks.append(y)

        logging.info(f"Built lists of length {len(names)} (receptors/ligands/names/pks).")
        return receptors, ligands, names, pks


    def _process_graphs(
            self, 
            receptors: List[str],
            ligands: List[str],
            names: List[str],
            pks: List[float],
            protein_cutoff: float=5.0, 
            pocket_cutoff: float=5.0, 
            spatial_cutoff: float=5.0
    ) -> None:
        success_count = 0
        # use ProcessPoolExecutor for CPU-bound tasks
        # pytorch objects can't be shared between processes, so we need to save them to disk in _process_single_graph() and then reload
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks with their indices
            pool_futures = [
                executor.submit(
                    self._process_single_graph, 
                    proteinpdb, 
                    ligandsdf, 
                    name, 
                    pk, 
                    protein_cutoff, 
                    pocket_cutoff, 
                    spatial_cutoff
                )
                for proteinpdb, ligandsdf, name, pk in zip(
                    receptors, ligands, names, pks
                )
            ]

            # Collect results as they complete
            try:
                for i, future in enumerate(as_completed(pool_futures)):
                    result = future.result()
                    success_count += result
                    if (i + 1) % 100 == 0 or (i + 1) == len(names):
                        logging.info(f"Processed {i + 1}/{len(names)} graphs")
            except KeyboardInterrupt:
                logging.error("KeyboardInterrupt received, terminating loading.")
                # let all workers terminate
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        
        # Log statistics
        failed_count = len(names) - success_count
        if failed_count > 0:
            logging.warning(
                f"Failed to process {failed_count}/{len(names)} graphs "
                f"(exceeded max_nodes or error)"
            )


    def _process_single_graph(
            self, 
            proteinpdb: str, 
            ligandsdf: str, 
            name: str, 
            pk: float, 
            protein_cutoff: float=5.0, 
            pocket_cutoff: float=5.0, 
            spatial_cutoff: float=5.0
    ) -> bool:
        # check if it is already processed
        file_path = self.processed_dir / f'{name}.pkl'
        if file_path.exists():
            logging.info(f"Graph {name} already processed, skipping.")
            return True
        
        try:
            graph = self._generate_raw_graph(
                Path(proteinpdb), 
                Path(ligandsdf), 
                name, 
                pk, 
                protein_cutoff, 
                pocket_cutoff, 
                spatial_cutoff
            )
            if graph is None:
                logging.error(f"Failed to generate graph for {name}")
                return False

            # Validate max_nodes constraint
            if graph.x.size(0) > self.max_nodes:
                logging.warning(
                    f"Graph {name} has {graph.x.size(0)} nodes > max_nodes {self.max_nodes}, skipping"
                )
                return False
            
            # Set graph properties
            graph.y = graph.y.reshape(-1)
            graph = preprocess_item(graph)
            with open(file_path, 'wb') as f:
                pickle.dump(graph, f)
            return True
        
        except KeyboardInterrupt:
            logging.error("KeyboardInterrupt received, terminating loading.")
            raise

        except Exception as e:
            logging.error(f"Error building {name}: {e}")
            return False


    def _generate_raw_graph(
            self,
            proteinpdb: Path, 
            ligandsdf: Path, 
            name: str, 
            pk: float, 
            protein_cutoff: float=5.0, 
            pocket_cutoff: float=5.0, 
            spatial_cutoff: float=5.0
    ) -> Optional[Data]:
        RDLogger.DisableLog('rdApp.*')

        if not (proteinpdb.is_file() and ligandsdf.is_file()):
            logging.error(f"{proteinpdb} or {ligandsdf} does not exist.")
            return None
        pocketpdb = proteinpdb.parent / (proteinpdb.name.rsplit('.', 1)[0] + '_pocket.pdb')
        pocketsdf = proteinpdb.parent / (proteinpdb.name.rsplit('.', 1)[0] + '_pocket.sdf')
        if not pocketpdb.is_file():
            pymol_pocket(proteinpdb, ligandsdf, pocketpdb)
        if not pocketsdf.is_file():
            obabel_pdb2mol(pocketpdb, pocketsdf)

        try:
            ligand = read_mol(ligandsdf)
            pocket = read_mol(pocketsdf)
            proinfo, liginfo = get_info(proteinpdb, ligandsdf)
            res = gen_feature(ligand, pocket, name)
            res['rfscore'] = RF_score(liginfo, proinfo)
            res['gbscore'] = GB_score(liginfo, proinfo)
            res['ecif'] = np.array(GetECIF(str(proteinpdb), str(ligandsdf)))
        except RuntimeError as e:
            logging.error(f"Failed to read molecules: {proteinpdb}, {pocketsdf}, {ligandsdf}")
            logging.error(e)
            return None

        ligand = (res['lc'], res['lf'], res['lei'], res['lea'])
        pocket = (res['pc'], res['pf'], res['pei'], res['pea'])
        try:
            raw = gen_graph(ligand, pocket, name, protein_cutoff=protein_cutoff, pocket_cutoff=pocket_cutoff, spatial_cutoff=spatial_cutoff)
        except ValueError as e:
            logging.error(f"{name}: Error gen_graph from raw feature {str(e)}")
            return None
        graph = to_pyg_graph(list(raw) + [res['rfscore'], res['gbscore'], res['ecif'], pk, name], frame=-1, rmsd_lig=0.0, rmsd_pro=0.0)

        return graph
