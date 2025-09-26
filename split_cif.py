#!/usr/bin/env python3
"""
cif_to_protein_ligand_pocket.py

Convert all .cif/.mmcif files in an input directory into:
 - <base>_protein.pdb                 (protein-only PDB)
 - <base>_ligand.sdf                  (ligand(s) only, converted to SDF)
 - <base>_protein_pocket.sdf          (protein pocket within 5.0 Å of any ligand atom, converted to SDF)

Dependencies:
  - biopython
  - openbabel (pybel)  [OpenBabel Python bindings]
Install with conda (example):
  conda install -c conda-forge biopython openbabel
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from Bio.PDB import MMCIFParser, PDBIO, Select, NeighborSearch
from Bio.PDB.Structure import Structure

try:
    from openbabel import pybel
except Exception as e:
    print("ERROR: openbabel (pybel) not available. Install OpenBabel (pybel) in your conda env.")
    raise

# --- Select classes for PDBIO ---
class ProteinSelect(Select):
    """Accept residues that are standard polymer residues (residue.id[0] == ' '). Exclude HOH."""
    def accept_residue(self, residue):
        if residue.id[0] != ' ':
            return False
        if residue.get_resname() == 'HOH':
            return False
        return True

class HeteroSelect(Select):
    """Accept hetero residues (residue.id[0] != ' ') excluding water."""
    def accept_residue(self, residue):
        if residue.id[0] == ' ':
            return False
        if residue.get_resname() == 'HOH':
            return False
        return True

class ResidueKeySelect(Select):
    """Accept residues whose (chain_id, resid) are in provided set."""
    def __init__(self, residue_keyset):
        self.keys = set(residue_keyset)
    def accept_residue(self, residue):
        chain = residue.get_parent()
        key = (chain.id, residue.id)   # residue.id is tuple: (hetflag, seqnum, icode)
        return key in self.keys

# --- helpers ---
def save_selection_to_pdb(structure: Structure, select_obj: Select, out_path: str):
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, select_obj)

def pdb_to_sdf_with_pybel(pdb_path: str, sdf_path: str, add_h: bool = True):
    """
    Convert a PDB file (possibly containing multiple small molecules / fragments) to multi-entry SDF.
    Uses pybel (OpenBabel) for bond perception and SDF writing.
    """
    # pybel.Outputfile handles append/overwrite logic
    outfile = pybel.Outputfile("sdf", sdf_path, overwrite=True)
    any_written = False
    for mol in pybel.readfile("pdb", pdb_path):
        if add_h:
            mol.addh()
        outfile.write(mol)
        any_written = True
    outfile.close()
    return any_written

def find_ligand_and_protein_atoms(structure: Structure) -> tuple[list, list]:
    ligand_atoms = []
    protein_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if residue.id[0] == ' ':
                        # polymeric residue (protein/nucleic acid); treat as protein
                        protein_atoms.append(atom)
                    else:
                        # hetero
                        if residue.get_resname() != 'HOH':  # skip water
                            ligand_atoms.append(atom)
    return ligand_atoms, protein_atoms

def compute_pocket_residue_keys(structure: Structure, ligand_atoms: list, protein_atoms: list, distance: float = 5.0) -> set:
    """
    Return a set of (chain_id, residue.id) keys for protein residues that have any atom within
    'distance' angstroms of any ligand atom.
    """
    if not ligand_atoms or not protein_atoms:
        return set()
    ns = NeighborSearch(protein_atoms)  # this NeighborSearch will only find neighbors among protein_atoms
    pocket_keys = set()
    for lat in ligand_atoms:
        neighs = ns.search(lat.coord, distance, level='A')  # returns atom objects (from protein_atoms)
        for atom in neighs:
            residue = atom.get_parent()
            chain = residue.get_parent()
            # ensure it's a protein residue
            if residue.id[0] == ' ' and residue.get_resname() != 'HOH':
                pocket_keys.add((chain.id, residue.id))
    return pocket_keys

# --- main processing for single file ---
def process_cif_file(cif_path: str, outdir: str, distance: float = 5.0):
    cif_path = Path(cif_path)
    base = cif_path.stem
    # allow ".mmcif" or ".cif" where stem may still include second extension; handle by stripping known suffixes
    if base.lower().endswith('.mmcif'):
        base = base[:-6]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    protein_pdb = outdir / f"{base}_protein.pdb"
    ligand_sdf = outdir / f"{base}_ligand.sdf"
    pocket_sdf = outdir / f"{base}_protein_pocket.sdf"
    # We'll write temporary PDBs for ligand and pocket before converting to SDF
    tmp_ligand_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp_ligand_pdb.close()
    tmp_pocket_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp_pocket_pdb.close()

    print(f"[INFO] Processing: {cif_path.name}")
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(base, str(cif_path))
    except Exception as e:
        print(f"[ERROR] Failed to parse {cif_path}: {e}")
        return

    # 1) write protein-only PDB
    try:
        save_selection_to_pdb(structure, ProteinSelect(), str(protein_pdb))
        print(f"  -> protein written: {protein_pdb}")
    except Exception as e:
        print(f"  [ERROR] Failed to write protein PDB for {cif_path.name}: {e}")

    # 2) write ligand-only PDB (all hetero except HOH) into tmp_ligand_pdb, then convert to sdf
    try:
        save_selection_to_pdb(structure, HeteroSelect(), tmp_ligand_pdb.name)
        # convert to SDF using pybel (will perform bond perception)
        written = pdb_to_sdf_with_pybel(tmp_ligand_pdb.name, str(ligand_sdf), add_h=True)
        if written:
            print(f"  -> ligand SDF written: {ligand_sdf}")
        else:
            # no hetero molecules found; remove any existing file and warn
            if Path(ligand_sdf).exists():
                Path(ligand_sdf).unlink()
            print(f"  -> no ligands (hetero residues) found in {cif_path.name}; ligand SDF not created.")
    except Exception as e:
        print(f"  [ERROR] Ligand extraction/conversion failed for {cif_path.name}: {e}")

    # 3) compute pocket residues (protein residues within 'distance' Å of any ligand atom)
    try:
        ligand_atoms, protein_atoms = find_ligand_and_protein_atoms(structure)
        pocket_keys = compute_pocket_residue_keys(structure, ligand_atoms, protein_atoms, distance=distance)
        if pocket_keys:
            # save pocket PDB (only those residues) then convert to SDF
            save_selection_to_pdb(structure, ResidueKeySelect(pocket_keys), tmp_pocket_pdb.name)
            written = pdb_to_sdf_with_pybel(tmp_pocket_pdb.name, str(pocket_sdf), add_h=True)
            if written:
                print(f"  -> pocket SDF written: {pocket_sdf} (distance = {distance} Å)")
            else:
                print(f"  -> pocket PDB created but OpenBabel wrote no molecules to SDF for {cif_path.name}.")
        else:
            print(f"  -> no pocket residues found within {distance} Å of ligands for {cif_path.name}; pocket SDF not created.")
    except Exception as e:
        print(f"  [ERROR] Pocket computation/conversion failed for {cif_path.name}: {e}")

    # cleanup tmp files
    try:
        os.unlink(tmp_ligand_pdb.name)
    except Exception:
        print(f"  [WARNING] Could not delete temporary file {tmp_ligand_pdb.name}")
    try:
        os.unlink(tmp_pocket_pdb.name)
    except Exception:
        print(f"  [WARNING] Could not delete temporary file {tmp_pocket_pdb.name}")

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Convert CIF/mmCIF complexes to protein PDB, ligand SDF, and pocket SDF.")
    parser.add_argument("input_dir", help="Input directory containing .cif or .mmcif files")
    parser.add_argument("output_dir", help="Directory where outputs will be written")
    parser.add_argument("--distance", type=float, default=5.0,
                        help="Distance in Angstrom to define binding pocket (default: 5.0 Å)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel worker processes (default: number of CPU cores)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input_dir '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)
    files = sorted(list(input_dir.glob("*.cif")) + list(input_dir.glob("*.mmcif")))
    if not files:
        print(f"No .cif or .mmcif files found in {input_dir}. Nothing to do.")
        return

    workers = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    print(f"Processing {len(files)} files with {workers} worker processes...")
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(process_cif_file, str(f), args.output_dir, args.distance): f.name for f in files}
        completed = 0
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                fut.result()
                completed += 1
            except Exception as exc:
                print(f"[ERROR] {fname} failed in worker: {exc}")
            if completed % 100 == 0:
                print(f"  {completed} files completed.")
    print("Done.")

if __name__ == "__main__":
    main()
