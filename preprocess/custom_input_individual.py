import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
from preprocess import gen_feature, gen_graph, to_pyg_graph, get_info, RF_score, GB_score
from ecif import GetECIF
from joblib import Parallel, delayed
from utils import read_mol, obabel_pdb2mol, pymol_pocket
import numpy as np
from rdkit import RDLogger
import pandas as pd


def process_one(proteinpdb: Path, ligandsdf: Path, name: str, pk: float, protein_cutoff, pocket_cutoff, spatial_cutoff):
    RDLogger.DisableLog('rdApp.*')

    if not (proteinpdb.is_file() and ligandsdf.is_file()):
        print(f"{proteinpdb} or {ligandsdf} does not exist.")
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
        print(proteinpdb, pocketsdf, ligandsdf, "Fail on reading molecule")
        print(e)
        return None

    ligand = (res['lc'], res['lf'], res['lei'], res['lea'])
    pocket = (res['pc'], res['pf'], res['pei'], res['pea'])
    try:
        raw = gen_graph(ligand, pocket, name, protein_cutoff=protein_cutoff, pocket_cutoff=pocket_cutoff, spatial_cutoff=spatial_cutoff)
    except ValueError as e:
        print(f"{name}: Error gen_graph from raw feature {str(e)}")
        return None
    graph = to_pyg_graph(list(raw) + [res['rfscore'], res['gbscore'], res['ecif'], pk, name], frame=-1, rmsd_lig=0.0, rmsd_pro=0.0)
    return graph


def build_lists(tsv_path: Path, data_dir: Path) -> tuple[list[str], list[str], list[str], list[float]]:
    """
    Read TSV and build receptors, ligands, names, pks lists.

    Parameters
    ----------
    tsv_path : Path
        Path to the TSV file (columns: drug, protein, y).
    data_dir : Path
        Directory containing files named <protein>_<drug>_model_0_protein.pdb and <protein>_<drug>_model_0_ligand.sdf.
    skip_missing : bool
        If True, skip rows where either file does not exist. If False, raise FileNotFoundError on first missing file.

    Returns
    -------
    receptors, ligands, names, pks
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype={"drug": str, "protein": str, "y": float})

    # Clean strings
    df["protein"] = df["protein"].astype(str).str.strip()
    df["drug"] = df["drug"].astype(str).str.strip()
    df["y"] = df["y"].astype(float)

    receptors: list[str] = []
    ligands: list[str] = []
    names: list[str] = []
    pks: list[float] = []

    for idx, row in df.iterrows():
        protein = row["protein"]
        drug = row["drug"]
        y = float(row["y"])

        base_name = f"{protein}_{drug}"
        pdb_path = data_dir / f"{base_name}_model_0_protein.pdb"
        sdf_path = data_dir / f"{base_name}_model_0_ligand.sdf"

        if not pdb_path.exists() or not sdf_path.exists():
            msg = f"Missing files for row {idx} ({base_name}): " \
                  f"{'PDB missing' if not pdb_path.exists() else ''} " \
                  f"{'SDF missing' if not sdf_path.exists() else ''}"
            raise FileNotFoundError(msg)

        receptors.append(str(pdb_path.resolve()))
        ligands.append(str(sdf_path.resolve()))
        names.append(base_name)
        pks.append(y)

    print(f"Built lists of length {len(names)} (receptors/ligands/names/pks).")
    return receptors, ligands, names, pks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv', '-t', required=True, type=Path, help="Path to the TSV file (columns: drug, protein, y).")
    parser.add_argument('--data-dir', '-d', required=True, type=Path, help="Directory containing PDB and SDF files.")
    parser.add_argument('--output', '-o', required=True, type=Path, help="Output directory for the processed graphs.")
    parser.add_argument('--njobs', '-n', type=int, default=-1)
    parser.add_argument('--protein_cutoff', type=float, default=5.)
    parser.add_argument('--pocket_cutoff', type=float, default=5.)
    parser.add_argument('--spatial_cutoff', type=float, default=5.)

    args = parser.parse_args()

    # convert to absolute paths
    tsv_path: Path = args.tsv.resolve()
    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output.resolve()
    receptors, ligands, names, pks = build_lists(tsv_path, data_dir)

    graphs = Parallel(n_jobs=args.njobs)(delayed(process_one)(Path(rec), Path(lig), name, pk, args.protein_cutoff, args.pocket_cutoff, args.spatial_cutoff) for rec, lig, name, pk in zip(tqdm(receptors), ligands, names, pks))

    for name, graph in zip(names, graphs):
        if graph is None:
            print(f"{name} failed.")
        else:
            output_path = output_dir / f'{name}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)
