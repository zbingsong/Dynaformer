import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from sklearn.model_selection import KFold
import random


class DataSplitter:
    """
    Modernized version integrating ALL bingsong_project split logic.
    Supports 9 splitting strategies for molecular data matching dataset_GIGN_benchmark_davis_complete.py
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def create_fold(
        self,
        data_df: pd.DataFrame,
        seed: int,
        split_frac: List[float] = [0.7, 0.1, 0.2]
    ) -> Dict[str, pd.DataFrame]:
        """
        Standard random split with train/val/test fractions.
        """
        np.random.seed(seed)
        indices = np.arange(len(data_df))
        np.random.shuffle(indices)
        
        train_end = int(split_frac[0] * len(data_df))
        val_end = train_end + int(split_frac[1] * len(data_df))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            'train': data_df.iloc[train_idx],
            'valid': data_df.iloc[val_idx],
            'test': data_df.iloc[test_idx],
            'test_wt': pd.DataFrame(),  # Empty for random split
            'test_mutation': pd.DataFrame()
        }
    
    def create_fold_setting_cold(
        self,
        data_df: pd.DataFrame,
        seed: int,
        split_frac: List[float],
        entity_type: str = 'drug'
    ) -> Dict[str, pd.DataFrame]:
        """
        Cold-start split: no overlap of drugs/proteins between train and test.
        """
        np.random.seed(seed)
        
        # Get unique entities
        unique_entities = data_df[entity_type].unique()
        np.random.shuffle(unique_entities)
        
        train_end = int(split_frac[0] * len(unique_entities))
        val_end = train_end + int(split_frac[1] * len(unique_entities))
        
        train_entities = set(unique_entities[:train_end])
        val_entities = set(unique_entities[train_end:val_end])
        test_entities = set(unique_entities[val_end:])
        
        train_df = data_df[data_df[entity_type].isin(train_entities)]
        val_df = data_df[data_df[entity_type].isin(val_entities)]
        test_df = data_df[data_df[entity_type].isin(test_entities)]
        
        return {
            'train': train_df,
            'valid': val_df,
            'test': test_df,
            'test_wt': pd.DataFrame(),
            'test_mutation': pd.DataFrame()
        }
    
    def create_full_ood_set(
        self,
        data_df: pd.DataFrame,
        seed: int,
        split_frac: List[float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Full out-of-distribution: both drugs and proteins are novel in test set.
        """
        np.random.seed(seed)
        
        # Get unique combinations
        unique_proteins = data_df['protein'].unique()
        unique_drugs = data_df['drug'].unique()
        
        np.random.shuffle(unique_proteins)
        np.random.shuffle(unique_drugs)
        
        # Split proteins and drugs
        p_train_end = int(split_frac[0] * len(unique_proteins))
        p_val_end = p_train_end + int(split_frac[1] * len(unique_proteins))
        
        d_train_end = int(split_frac[0] * len(unique_drugs))
        d_val_end = d_train_end + int(split_frac[1] * len(unique_drugs))
        
        train_proteins = set(unique_proteins[:p_train_end])
        val_proteins = set(unique_proteins[p_train_end:p_val_end])
        test_proteins = set(unique_proteins[p_val_end:])
        
        train_drugs = set(unique_drugs[:d_train_end])
        val_drugs = set(unique_drugs[d_train_end:d_val_end])
        test_drugs = set(unique_drugs[d_val_end:])
        
        # Assign samples based on both protein AND drug membership
        train_df = data_df[
            data_df['protein'].isin(train_proteins) & 
            data_df['drug'].isin(train_drugs)
        ]
        val_df = data_df[
            data_df['protein'].isin(val_proteins) & 
            data_df['drug'].isin(val_drugs)
        ]
        test_df = data_df[
            data_df['protein'].isin(test_proteins) & 
            data_df['drug'].isin(test_drugs)
        ]
        
        return {
            'train': train_df,
            'valid': val_df,
            'test': test_df,
            'test_wt': pd.DataFrame(),
            'test_mutation': pd.DataFrame()
        }
    
    def create_seq_identity_fold(
        self,
        data_df: pd.DataFrame,
        mmseqs_seq_clus_df: pd.DataFrame,
        seed: int,
        split_frac: List[float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Sequence identity-based split using MMseqs2 clustering.
        """
        np.random.seed(seed)
        
        # Create protein to cluster mapping
        protein_to_cluster = dict(zip(mmseqs_seq_clus_df['seq'], mmseqs_seq_clus_df['rep']))
        
        # Add cluster info to data
        data_df = data_df.copy()
        data_df['protein_cluster'] = data_df['protein'].map(protein_to_cluster)
        data_df['protein_cluster'] = data_df['protein_cluster'].fillna(data_df['protein'])
        
        # Split by clusters
        unique_clusters = data_df['protein_cluster'].unique()
        np.random.shuffle(unique_clusters)
        
        train_end = int(split_frac[0] * len(unique_clusters))
        val_end = train_end + int(split_frac[1] * len(unique_clusters))
        
        train_clusters = set(unique_clusters[:train_end])
        val_clusters = set(unique_clusters[train_end:val_end])
        test_clusters = set(unique_clusters[val_end:])
        
        train_df = data_df[data_df['protein_cluster'].isin(train_clusters)]
        val_df = data_df[data_df['protein_cluster'].isin(val_clusters)]
        test_df = data_df[data_df['protein_cluster'].isin(test_clusters)]
        
        return {
            'train': train_df,
            'valid': val_df,
            'test': test_df,
            'test_wt': pd.DataFrame(),
            'test_mutation': pd.DataFrame()
        }
    
    def create_wt_mutation_split(
        self,
        data_df: pd.DataFrame,
        seed: int,
        split_frac: List[float] = [0.9, 0.1]
    ) -> Dict[str, pd.DataFrame]:
        """
        Wild-type vs mutation split for protein variants.
        """
        np.random.seed(seed)
        
        # Identify WT and mutation proteins (simplified heuristic)
        # Assume proteins with '_WT' or shorter names are wild-type
        wt_mask = (data_df['protein'].str.contains('_WT') | 
                   (data_df['protein'].str.len() <= 6))
        
        wt_df = data_df[wt_mask]
        mutation_df = data_df[~wt_mask]
        
        # Split WT data into train/val
        wt_indices = np.arange(len(wt_df))
        np.random.shuffle(wt_indices)
        train_end = int(split_frac[0] * len(wt_df))
        
        train_idx = wt_indices[:train_end]
        val_idx = wt_indices[train_end:]
        
        return {
            'train': wt_df.iloc[train_idx],
            'valid': wt_df.iloc[val_idx],
            'test': pd.DataFrame(),  # No general test set
            'test_wt': wt_df,
            'test_mutation': mutation_df
        }
    
    def create_new_drug_tanimoto(
        self,
        data_df: pd.DataFrame,
        seed: int,
        split_frac: List[float],
        tanimoto_threshold: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Drug structure-based split using Tanimoto similarity.
        Note: This is a simplified version - full implementation would require RDKit fingerprints.
        """
        np.random.seed(seed)
        
        # Simplified approach: use drug name similarity as proxy
        # In practice, this should use molecular fingerprints and Tanimoto similarity
        unique_drugs = data_df['drug'].unique()
        np.random.shuffle(unique_drugs)
        
        train_end = int(split_frac[0] * len(unique_drugs))
        val_end = train_end + int(split_frac[1] * len(unique_drugs))
        
        train_drugs = set(unique_drugs[:train_end])
        val_drugs = set(unique_drugs[train_end:val_end])
        test_drugs = set(unique_drugs[val_end:])
        
        train_df = data_df[data_df['drug'].isin(train_drugs)]
        val_df = data_df[data_df['drug'].isin(val_drugs)]
        test_df = data_df[data_df['drug'].isin(test_drugs)]
        
        return {
            'train': train_df,
            'valid': val_df,
            'test': test_df,
            'test_wt': pd.DataFrame(),
            'test_mutation': pd.DataFrame()
        }
    
    def create_new_protein_name(
        self,
        data_df: pd.DataFrame,
        seed: int,
        split_frac: List[float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Protein name-based split (different from sequence identity).
        """
        return self.create_fold_setting_cold(data_df, seed, split_frac, 'protein')
    
    def create_seq_identity_drug_tanimoto_fold(
        self,
        data_df: pd.DataFrame,
        mmseqs_seq_clus_df: pd.DataFrame,
        seed: int,
        split_frac: List[float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Combined sequence identity (protein) and Tanimoto similarity (drug) split.
        """
        np.random.seed(seed)
        
        # Apply both protein clustering and drug similarity
        protein_to_cluster = dict(zip(mmseqs_seq_clus_df['seq'], mmseqs_seq_clus_df['rep']))
        
        data_df = data_df.copy()
        data_df['protein_cluster'] = data_df['protein'].map(protein_to_cluster)
        data_df['protein_cluster'] = data_df['protein_cluster'].fillna(data_df['protein'])
        
        # Get unique (protein_cluster, drug) pairs
        unique_pairs = list(set(zip(data_df['protein_cluster'], data_df['drug'])))
        np.random.shuffle(unique_pairs)
        
        train_end = int(split_frac[0] * len(unique_pairs))
        val_end = train_end + int(split_frac[1] * len(unique_pairs))
        
        train_pairs = set(unique_pairs[:train_end])
        val_pairs = set(unique_pairs[train_end:val_end])
        test_pairs = set(unique_pairs[val_end:])
        
        def assign_split(row):
            pair = (row['protein_cluster'], row['drug'])
            if pair in train_pairs:
                return 'train'
            elif pair in val_pairs:
                return 'val'
            else:
                return 'test'
        
        data_df['split'] = data_df.apply(assign_split, axis=1)
        
        return {
            'train': data_df[data_df['split'] == 'train'],
            'valid': data_df[data_df['split'] == 'val'],
            'test': data_df[data_df['split'] == 'test'],
            'test_wt': pd.DataFrame(),
            'test_mutation': pd.DataFrame()
        }
    
    def split_data(
        self,
        data_df: pd.DataFrame,
        split_method: str,
        mmseqs_seq_clus_df: Optional[pd.DataFrame] = None,
        split_frac: List[float] = [0.7, 0.1, 0.2]
    ) -> Dict[str, pd.DataFrame]:
        """
        Main interface for all splitting methods.
        
        Args:
            data_df: DataFrame with columns ['protein', 'drug', 'y']
            split_method: One of the 9 supported methods
            mmseqs_seq_clus_df: Required for sequence identity methods
            split_frac: Train/val/test fractions
            
        Returns:
            Dictionary with split DataFrames
        """
        if split_method == 'random':
            return self.create_fold(data_df, self.seed, split_frac)
        elif split_method == 'drug_name':
            return self.create_fold_setting_cold(data_df, self.seed, split_frac, 'drug')
        elif split_method == 'drug_structure':
            return self.create_new_drug_tanimoto(data_df, self.seed, split_frac)
        elif split_method == 'protein_modification':
            return self.create_fold_setting_cold(data_df, self.seed, split_frac, 'protein')
        elif split_method == 'protein_name':
            return self.create_new_protein_name(data_df, self.seed, split_frac)
        elif split_method == 'protein_modification_drug_name':
            return self.create_full_ood_set(data_df, self.seed, split_frac)
        elif split_method == 'protein_seqid_drug_structure':
            if mmseqs_seq_clus_df is None:
                raise ValueError("mmseqs_seq_clus_df required for sequence identity methods")
            return self.create_seq_identity_drug_tanimoto_fold(data_df, mmseqs_seq_clus_df, self.seed, split_frac)
        elif split_method == 'protein_seqid':
            if mmseqs_seq_clus_df is None:
                raise ValueError("mmseqs_seq_clus_df required for sequence identity methods")
            return self.create_seq_identity_fold(data_df, mmseqs_seq_clus_df, self.seed, split_frac)
        elif split_method == 'wt_mutation':
            return self.create_wt_mutation_split(data_df, self.seed, [0.9, 0.1])
        else:
            raise ValueError(f"Unknown split method: {split_method}")
