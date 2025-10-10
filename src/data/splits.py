import pandas as pd
import numpy as np
import random
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from typing import List, Tuple, Dict, Optional, Set


class DataSplitter(object):
    def __init__(
            self, 
            data_df: pd.DataFrame, 
            mmseqs_seq_clus_df: Optional[pd.DataFrame]=None,
            seed: int=42,
    ):
        self.data_df = data_df
        self.mmseqs_seq_clus_df = mmseqs_seq_clus_df
        self.seed = seed

    def generate_split_indices(self, split_method: str, split_frac: Tuple[float, float, float]) -> Dict[str, pd.Index]:
        if split_method == 'random':
            split_indices_dict = self.create_random_fold(self.data_df, self.seed, split_frac)
        elif split_method == 'drug_name':
            split_indices_dict = self.create_fold_setting_cold(self.data_df, self.seed, split_frac, 'drug')
        elif split_method == 'drug_structure':
            split_indices_dict = self.create_new_drug_tanimoto(self.data_df, self.seed, split_frac)
        elif split_method == 'protein_modification':
            split_indices_dict = self.create_fold_setting_cold(self.data_df, self.seed, split_frac, 'protein')
        elif split_method == 'protein_name':
            split_indices_dict = self.create_new_protein_name(self.data_df, self.seed, split_frac)
        elif split_method == 'protein_modification_drug_name':
            split_indices_dict = self.create_full_ood_set(self.data_df, self.seed, split_frac)
        elif split_method == 'protein_seqid_drug_structure':
            split_indices_dict = self.create_seq_identity_drug_tanimoto_fold(self.data_df, self.mmseqs_seq_clus_df, self.seed, split_frac)
        elif split_method == 'protein_seqid':
            split_indices_dict = self.create_seq_identity_fold(self.data_df, self.mmseqs_seq_clus_df, self.seed, split_frac)
        elif split_method == 'wt_mutation':
            split_indices_dict = self.create_wt_mutation_split(self.data_df, self.seed, split_frac)
        else:
            raise ValueError("Unknown split method: {}".format(split_method))
        return split_indices_dict


    def _smiles_to_fingerprint(self, smiles: str, radius: int = 2, nBits: int = 2048):
        """Convert SMILES to Morgan fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)


    def _calculate_tanimoto_similarity(self, fp1, fp2):
        """Calculate Tanimoto similarity between two fingerprints"""
        return DataStructs.TanimotoSimilarity(fp1, fp2)


    def _get_similarity_matrix(self, fingerprints: List):
        """Calculate pairwise Tanimoto similarity matrix"""
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self._calculate_tanimoto_similarity(fingerprints[i], fingerprints[j])
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
        
        return similarity_matrix
    

    def _parse_mmseqs_cluster_res(self, mmseqs_seq_clus_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        clus2seq, seq2clus = {}, {}
        for rep, sdf in mmseqs_seq_clus_df.groupby('rep'):
            for seq in sdf['seq']:
                if rep not in clus2seq:
                    clus2seq[rep] = []
                clus2seq[rep].append(seq)
                seq2clus[seq] = rep
        return seq2clus, clus2seq


    def _create_cluster_split(
            self, 
            df: pd.DataFrame, 
            seq2clus: Dict[str, str], 
            clus2seq: Dict[str, List[str]], 
            to_use: Set[str], 
            split_size_fp: float, 
            min_clus_in_split: int, 
            fold_seed: int
    ):
        _rng = np.random.RandomState(fold_seed)
        data = df.copy()
        all_prot = set(seq2clus.keys())
        used = all_prot.difference(to_use)
        split = None
        while True:
            p = _rng.choice(sorted(to_use))
            c = seq2clus[p]
            members = set(clus2seq[c])
            members = members.difference(used)
            if len(members) == 0:
                continue
            # ensure that at least min_fam_in_split families in each split
            max_clust_size = int(np.ceil(split_size_fp / min_clus_in_split))
            sel_prot = list(members)[:max_clust_size]
            sel_df = data[data['protein'].isin(sel_prot)]
            split = sel_df if split is None else pd.concat([split, sel_df])
            to_use = to_use.difference(members)
            used = used.union(members)
            if len(split) >= split_size_fp:
                break
        split = split.reset_index(drop=True)
        return split, to_use


    def create_random_fold(
            self, 
            df: pd.DataFrame, 
            fold_seed: int, 
            frac: Tuple[float, float, float]
    ) -> Dict[str, pd.Index]:
        """
        Create train/valid/test folds by random splitting.
        Adapted from: https://github.com/mims-harvard/TDC/blob/2d4fb74ac00e88986306b2b12ffdb3be87418719/tdc/utils.py#L375
        """
        _, val_frac, test_frac = frac
        test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
        test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") 
                            | test['protein'].str.contains("_itd") 
                            | test['protein'].str.contains("abl1_p") 
                            | test['protein'].str.contains("s808g")]
        test_wt = test[~test.index.isin(test_mutation.index)]
        
        train_val = df[~df.index.isin(test.index)]
        val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]

        return {'train': train.index,
                'valid': val.index,
                'test': test.index,
                'test_wt': test_wt.index,
                'test_mutation': test_mutation.index}


    def create_new_drug_tanimoto(
            self, 
            df: pd.DataFrame, 
            fold_seed: int, 
            frac: Tuple[float, float, float]
    ) -> Dict[str, pd.Index]:
        """
        Create train/valid/test folds by tanimoto similarity of drugs. Similarity of 0.5 is used as the threshold.
        """
        all_smiles = list(sorted(set(df['compound_iso_smiles'])))
        val_size = int(len(all_smiles) * frac[1])
        test_size = int(len(all_smiles) * frac[2])

        all_fp = []
        for smiles in all_smiles:
            fp = self._smiles_to_fingerprint(smiles)
            if fp is not None:
                all_fp.append(fp)

        similarity_matrix_all_fp = self._get_similarity_matrix(all_fp)

        selected_index = []
        for i in range(len(all_fp)):
            if len(similarity_matrix_all_fp[:, i][similarity_matrix_all_fp[:, i] >= 0.5]) < 2:
                selected_index.append(i)
        
        random.seed(fold_seed)
        shuffled_indices = selected_index.copy()
        random.shuffle(shuffled_indices)

        val_indices = shuffled_indices[:val_size]
        test_indices = shuffled_indices[val_size:val_size + test_size]

        val_smiles = [all_smiles[i] for i in val_indices]
        test_smiles = [all_smiles[i] for i in test_indices]
        train_smiles = [all_smiles[i] for i in range(len(all_smiles)) if i not in val_indices and i not in test_indices]
        
        test = df[df['compound_iso_smiles'].isin(test_smiles)]
        test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") 
                            | test['protein'].str.contains("_itd") 
                            | test['protein'].str.contains("abl1_p") 
                            | test['protein'].str.contains("s808g")]
        test_wt = test[~test.index.isin(test_mutation.index)]

        val = df[df['compound_iso_smiles'].isin(val_smiles)]
        train = df[df['compound_iso_smiles'].isin(train_smiles)]

        return {'train': train.index,
                'valid': val.index,
                'test': test.index,
                'test_wt': test_wt.index,
                'test_mutation': test_mutation.index}
        

    def create_wt_mutation_split(
            self, 
            df: pd.DataFrame, 
            fold_seed: int, 
            frac: Tuple[float, float, float]
    ) -> Dict[str, pd.Index]:
        _, val_frac, _ = frac
        test = df[(df['protein'].str.contains(f"_[a-z][0-9]") 
                | df['protein'].str.contains(f"_itd") 
                | df['protein'].str.contains(f"abl1_p") 
                | df['protein'].str.contains("s808g"))]
        train_val = df[~df.index.isin(test.index)]
        test_frac = len(test) / len(df)
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed)
        train = train_val[~train_val.index.isin(val.index)]

        return {'train': train.index,
                'valid': val.index,
                'test': test.index,
                'test_wt': pd.Index([]),
                'test_mutation': test.index}


    def create_fold_setting_cold(
            self, 
            df: pd.DataFrame, 
            fold_seed: int, 
            frac: Tuple[float, float, float], 
            entity: str
    ) -> Dict[str, pd.Index]:
        """
        Create train/valid/test folds by drug/protein-wise splitting.
        Adapted from: https://github.com/mims-harvard/TDC/blob/2d4fb74ac00e88986306b2b12ffdb3be87418719/tdc/utils.py#L388
        """
        _, val_frac, test_frac = frac
        gene_drop = df[entity].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values

        test = df[df[entity].isin(gene_drop)]
        test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
        test_wt = test[~test.index.isin(test_mutation.index)]

        train_val = df[~df[entity].isin(gene_drop)]

        gene_drop_val = train_val[entity].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
        val = train_val[train_val[entity].isin(gene_drop_val)]
        train = train_val[~train_val[entity].isin(gene_drop_val)]

        return {'train': train.index,
                'valid': val.index,
                'test': test.index,
                'test_wt': test_wt.index,
                'test_mutation': test_mutation.index}


    def create_new_protein_name(
            self, 
            df: pd.DataFrame, 
            fold_seed: int, 
            frac: Tuple[float, float, float]
    ) -> Dict[str, pd.Index]:
        """
        Create train/valid/test folds by drug/protein-wise splitting.
        Adapted from: https://github.com/mims-harvard/TDC/blob/2d4fb74ac00e88986306b2b12ffdb3be87418719/tdc/utils.py#L388
        """
        protein_mutation = ['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'gcn2', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret'] 

        df_mutation = df[df['protein'].str.contains("_[a-z][0-9]") | df['protein'].str.contains("itd") | df['protein'].str.contains("abl1_p")]
        df_wt = df[~df.index.isin(df_mutation.index)]

        _, val_frac, test_frac = frac
        test_sample_protein = df_wt['protein'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values

        patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in test_sample_protein]
        regex = '|'.join(patterns)
        test = df[df['protein'].str.contains(regex, regex=True)]

        test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
        test_wt = test[~test.index.isin(test_mutation.index)]

        df_wt_train_val = df_wt[~df_wt['protein'].isin(test_sample_protein)]
        val_sample_protein = df_wt_train_val['protein'].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
        patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in val_sample_protein]
        regex = '|'.join(patterns)
        val = df[df['protein'].str.contains(regex, regex=True)]

        train_sample_protein = df_wt_train_val[~df_wt_train_val['protein'].isin(val_sample_protein)]['protein'].drop_duplicates().values
        patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in train_sample_protein]
        regex = '|'.join(patterns)
        train = df[df['protein'].str.contains(regex, regex=True)]
        
        # Verify no overlap between splits
        assert len(set(train['protein']) & set(val['protein'])) == 0, "Overlap between train and validation proteins"
        assert len(set(test['protein']) & set(val['protein'])) == 0, "Overlap between test and validation proteins"
        assert len(set(train['protein']) & set(test['protein'])) == 0, "Overlap between train and test proteins"

        return {'train': train.index,
                'valid': val.index,
                'test': test.index,
                'test_wt': test_wt.index,
                'test_mutation': test_mutation.index}


    def create_full_ood_set(
            self, 
            df: pd.DataFrame, 
            fold_seed: int, 
            frac: Tuple[float, float, float]
    ) -> Dict[str, pd.Index]:
        """
        Create train/valid/test folds such that drugs and proteins are
        not overlapped in train and test sets. Train and valid may share
        drugs and proteins (random split).
        """
        _, val_frac, test_frac = frac
        test_drugs = df['drug'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values
        test_prots = df['protein'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values

        test = df[(df['drug'].isin(test_drugs)) & (df['protein'].isin(test_prots))]
        test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
        test_wt = test[~test.index.isin(test_mutation.index)]

        train_val = df[(~df['drug'].isin(test_drugs)) & (~df['protein'].isin(test_prots))]
        val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed)
        train = train_val[~train_val.index.isin(val.index)]

        return {'train': train.index,
                'valid': val.index,
                'test': test.index,
                'test_wt': test_wt.index,
                'test_mutation': test_mutation.index}


    def create_seq_identity_fold(
            self, 
            df: pd.DataFrame, 
            mmseqs_seq_clus_df: Optional[pd.DataFrame], 
            fold_seed: int, 
            frac: Tuple[float, float, float], 
            min_clus_in_split: int=5
    ) -> Dict[str, pd.Index]:
        """
        Adapted from: https://github.com/drorlab/atom3d/blob/master/atom3d/splits/sequence.py
        Clusters are selected randomly into validation and test sets,
        but to ensure that there is some diversity in each set
        (i.e. a split does not consist of a single sequence cluster), a minimum number of clusters in each split is enforced.
        Some data examples may be removed in order to satisfy this constraint.
        """
        assert mmseqs_seq_clus_df is not None, "mmseqs_seq_clus_df is required for sequence identity splitting"
        seq2clus, clus2seq = self._parse_mmseqs_cluster_res(mmseqs_seq_clus_df)
        _, val_frac, test_frac = frac
        test_size, val_size = len(df) * test_frac, len(df) * val_frac
        to_use = set(seq2clus.keys())

        val_df, to_use = self._create_cluster_split(df, seq2clus, clus2seq, to_use, val_size, min_clus_in_split, fold_seed)
        test_df, to_use = self._create_cluster_split(df, seq2clus, clus2seq, to_use, test_size, min_clus_in_split, fold_seed)
        train_df = df[df['protein'].isin(to_use)].reset_index(drop=True)
        train_df['split'] = 'train'
        val_df['split'] = 'valid'
        test_df['split'] = 'test'

        test_df_mutation = test_df[test_df['protein'].str.contains("_[a-z][0-9]") | test_df['protein'].str.contains("_itd") | test_df['protein'].str.contains("abl1_p") | test_df['protein'].str.contains("s808g")]
        test_df_wt = test_df[~test_df.index.isin(test_df_mutation.index)]

        assert len(set(train_df['protein']) & set(val_df['protein'])) == 0
        assert len(set(test_df['protein']) & set(val_df['protein'])) == 0
        assert len(set(train_df['protein']) & set(test_df['protein'])) == 0

        return {'train': train_df.index,
                'valid': val_df.index,
                'test': test_df.index,
                'test_wt': test_df_wt.index,
                'test_mutation': test_df_mutation.index}


    def create_seq_identity_drug_tanimoto_fold(
            self, 
            df: pd.DataFrame, 
            mmseqs_seq_clus_df: Optional[pd.DataFrame], 
            fold_seed: int, 
            frac: Tuple[float, float, float], 
            min_clus_in_split: int=5
    ) -> Dict[str, pd.Index]:
        """
        Adapted from: https://github.com/drorlab/atom3d/blob/master/atom3d/splits/sequence.py
        Clusters are selected randomly into validation and test sets,
        but to ensure that there is some diversity in each set
        (i.e. a split does not consist of a single sequence cluster), a minimum number of clusters in each split is enforced.
        Some data examples may be removed in order to satisfy this constraint.
        """
        assert mmseqs_seq_clus_df is not None, "mmseqs_seq_clus_df is required for sequence identity splitting"
        seq2clus, clus2seq = self._parse_mmseqs_cluster_res(mmseqs_seq_clus_df)
        _, val_frac, test_frac = frac
        test_size, val_size = len(df) * test_frac, len(df) * val_frac
        to_use = set(seq2clus.keys())

        val_df, to_use = self._create_cluster_split(df, seq2clus, clus2seq, to_use, val_size, min_clus_in_split, fold_seed)
        test_df, to_use = self._create_cluster_split(df, seq2clus, clus2seq, to_use, test_size, min_clus_in_split, fold_seed)
        train_df = df[df['protein'].isin(to_use)].reset_index(drop=True)
        train_df['split'] = 'train'
        val_df['split'] = 'valid'
        test_df['split'] = 'test'
        
        ### here we add the drug tanimoto similarity filtering
        all_smiles = list(sorted(set(df['compound_iso_smiles'])))
        val_size = int(len(all_smiles) * frac[1])
        test_size = int(len(all_smiles) * frac[2])

        all_fp = []
        for smiles in all_smiles:
            fp = self._smiles_to_fingerprint(smiles)
            if fp is not None:
                all_fp.append(fp)

        similarity_matrix_all_fp = self._get_similarity_matrix(all_fp)

        selected_index = []
        for i in range(len(all_fp)):
            if len(similarity_matrix_all_fp[:, i][similarity_matrix_all_fp[:, i] >= 0.5]) < 2:
                selected_index.append(i)

        random.seed(fold_seed)
        shuffled_indices = selected_index.copy()
        random.shuffle(shuffled_indices)

        val_indices = shuffled_indices[:val_size]
        test_indices = shuffled_indices[val_size:val_size + test_size]

        val_smiles = [all_smiles[i] for i in val_indices]
        test_smiles = [all_smiles[i] for i in test_indices]
        train_smiles = [all_smiles[i] for i in range(len(all_smiles)) if i not in val_indices and i not in test_indices]
        
        test_df = test_df[test_df['compound_iso_smiles'].isin(test_smiles)]
        val_df = val_df[val_df['compound_iso_smiles'].isin(val_smiles)]
        train_df = train_df[train_df['compound_iso_smiles'].isin(train_smiles)]
        
        test_df_mutation = test_df[test_df['protein'].str.contains("_[a-z][0-9]") | test_df['protein'].str.contains("_itd") | test_df['protein'].str.contains("abl1_p") | test_df['protein'].str.contains("s808g")]
        test_df_wt = test_df[~test_df.index.isin(test_df_mutation.index)]

        assert len(set(train_df['protein']) & set(val_df['protein'])) == 0
        assert len(set(test_df['protein']) & set(val_df['protein'])) == 0
        assert len(set(train_df['protein']) & set(test_df['protein'])) == 0

        return {'train': train_df.index,
                'valid': val_df.index,
                'test': test_df.index,
                'test_wt': test_df_wt.index,
                'test_mutation': test_df_mutation.index}
