#!/usr/bin/env python3
"""
cif_to_pymol_rdkit.py

Use PyMOL's Python API (pymol.cmd) to split CIF/mmCIF into protein PDB, ligand PDB(s),
and pocket PDB, then use RDKit to add hydrogens and write ligand/pocket SDFs.

Usage:
    python cif_to_pymol_rdkit.py /path/to/cifs /path/to/output --distance 5.0

Dependencies:
    - PyMOL (pymol package; must be installed and importable)
    - RDKit (rdkit.Chem)
"""

import argparse
from pathlib import Path
import sys
import tempfile
import shutil
import re
from typing import Optional
from joblib import Parallel, delayed

import pandas as pd
from pymol import cmd

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

# ---------- helper functions ----------
def pymol_extract(cif_path: Path, tmpdir: Path):
    """
    Use pymol.cmd to load cif_path and save:
      - protein PDB -> tmpdir/protein_raw.pdb (polymer, not HOH)
      - ligand PDB  -> tmpdir/ligand_raw.pdb (hetero, not HOH; resn forced to LIG)
    Returns tuple (protein_pdb_path, ligand_pdb_path) -- any may be None if not produced.
    """
    obj = "myobj"
    # names in tmpdir
    protein_raw = tmpdir / "protein_raw.pdb"
    ligand_raw = tmpdir / "ligand_raw.pdb"

    # Load (discrete=0 to allow editing)
    cmd.reinitialize()  # clear previous state
    # use absolute path to avoid cwd issues
    cmd.load(str(cif_path.resolve()), obj, discrete=0)

    # Select ligands (hetero, exclude water) and force resname to 'LIG'
    cmd.select("ligands", f"{obj} and not polymer and not resn HOH")
    # If no hetero present, selection is empty (PyMOL won't fail)
    # Force the 3-letter resname for safe PDB output
    cmd.alter("ligands", "resn='LIG'")
    # Save ligand PDB (if any)
    cmd.save(str(ligand_raw), "ligands")

    # Save protein (polymer, not water)
    cmd.select("protein_sel", f"{obj} and polymer and not resn HOH")
    cmd.save(str(protein_raw), "protein_sel")

    return (protein_raw if protein_raw.exists() else None,
            ligand_raw if ligand_raw.exists() else None)


def rdkit_add_h_and_write_sdf(sdf_template_path: str, pdb_path: str, out_sdf: str) -> None:
    # load template (with bond info)
    templ_supplier = Chem.SDMolSupplier(sdf_template_path, removeHs=True)
    template = next((m for m in templ_supplier if m is not None), None)
    if template is None:
        raise RuntimeError("failed to read template.sdf")
    template.RemoveAllConformers()
    template.UpdatePropertyCache(strict=False)

    # load pdb (do not force RDKit to sanitize / re-perceive bonds yet)
    pdb = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    if pdb is None:
        raise RuntimeError("failed to read input.pdb")
    pdb.UpdatePropertyCache(strict=False)

    # Option A: use AssignBondOrdersFromTemplate (recommended if ordering differs)
    try:
        newmol = AllChem.AssignBondOrdersFromTemplate(template, pdb)
        # print("AssignBondOrdersFromTemplate succeeded")
        # reassign stereochemistry/aromaticity
        newmol = Chem.AddHs(newmol, addCoords=True)
        Chem.SanitizeMol(newmol)
        # Chem.AssignStereochemistry(newmol, cleanIt=True, force=True)
        Chem.MolToMolFile(newmol, out_sdf, forceV3000=True)
        print("Wrote", out_sdf, "(AssignBondOrdersFromTemplate succeeded)")
        return
    except Exception as e:
        print("AssignBondOrdersFromTemplate failed:", e)

    # Option B: rdFMCS mapping fallback (map indices and copy coords)
    print("Trying MCS-based fallback mapping...")
    mcs = rdFMCS.FindMCS([template, pdb], timeout=10)
    if mcs.canceled:
        raise RuntimeError("MCS timed out/unhelpful")
    mcs_q = Chem.MolFromSmarts(mcs.smartsString)
    t_match = template.GetSubstructMatch(mcs_q)
    p_match = pdb.GetSubstructMatch(mcs_q)
    if not t_match or not p_match:
        raise RuntimeError("MCS matching failed; molecules may differ in chemistry/tautomers")

    # Build a conformer for the template, copying mapped atom coords from pdb
    new_conf = Chem.Conformer(template.GetNumAtoms())
    p_conf = pdb.GetConformer()
    # copy mapped atoms
    for ti, pi in zip(t_match, p_match):
        new_conf.SetAtomPosition(ti, p_conf.GetAtomPosition(pi))
    # for unmapped atoms, you can either keep template coords or set to the mapped-atom centroid:
    for i in range(template.GetNumAtoms()):
        if not any(i == tm for tm in t_match):
            raise RuntimeError("Unmapped atoms present; please ensure full mapping or use a different method")

    template.AddConformer(new_conf, assignId=True)
    Chem.SanitizeMol(template)
    Chem.AddHs(template)
    Chem.MolToMolFile(template, out_sdf)
    print("Wrote", out_sdf, "(MCS fallback — mapped coords copied)")


def assign_bonds_from_smiles_robust(smiles: str, pdb_path: str, out_sdf: str, add_hydrogens=True):
    """
    More robust version that handles potential atom ordering issues.
    Uses 3D coordinate matching to align PDB and SMILES structures.
    
    Args:
        ligand_pdb_path: Path to ligand PDB file
        smiles: SMILES string of the ligand
        output_sdf_path: Path to output SDF file
        add_hydrogens: Whether to add hydrogens (default: True)
        
    Returns:
        RDKit Mol object with assigned bonds and coordinates
    """
    # Load PDB
    # print(f"Reading ligand PDB from {pdb_path}...")
    pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    # pdb_mol = read_pdb_coordinates_only(ligand_pdb_path)
    if pdb_mol is None:
        raise ValueError(f"Could not read PDB from {pdb_path}")
    
    # Create template from SMILES
    # print(f"Creating template from SMILES: {smiles}")
    template_mol = Chem.MolFromSmiles(smiles)
    if template_mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Add 3D coordinates to template (for matching)
    template_mol = Chem.AddHs(template_mol)
    AllChem.EmbedMolecule(template_mol, randomSeed=42)
    
    # Remove hydrogens for matching
    pdb_mol_no_h = Chem.RemoveHs(pdb_mol)
    template_no_h = Chem.RemoveHs(template_mol)

    mcs = rdFMCS.FindMCS([template_no_h, pdb_mol_no_h])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match1 = template_no_h.GetSubstructMatch(mcs_mol)
    match2 = pdb_mol_no_h.GetSubstructMatch(mcs_mol)
    if not match1 or not match2:
        raise RuntimeError("MCS matching failed")

    new_conf = Chem.Conformer(template_no_h.GetNumAtoms())
    p_conf = pdb_mol_no_h.GetConformer()
    # copy mapped atoms
    for ti, pi in zip(match1, match2):
        new_conf.SetAtomPosition(ti, p_conf.GetAtomPosition(pi))
    # for unmapped atoms, you can either keep template coords or set to the mapped-atom centroid:
    for i in range(template_no_h.GetNumAtoms()):
        if not any(i == tm for tm in match1):
            raise RuntimeError("Unmapped atoms present; please ensure full mapping or use a different method")

    template_no_h.AddConformer(new_conf, assignId=True)
    Chem.SanitizeMol(template_no_h)
    Chem.AddHs(template_no_h)

    # print('MCS matching results: ', match1, match2)

    match = pdb_mol_no_h.GetSubstructMatch(template_no_h)
    reordered = Chem.RenumberAtoms(pdb_mol_no_h, match)
    
    # print("Attempting to assign bonds with atom reordering...")
    # This will try to match atoms by 3D proximity if direct assignment fails
    try:
        mol_with_bonds = AllChem.AssignBondOrdersFromTemplate(template_no_h, reordered)
    except ValueError:
        # If direct assignment fails, try with sanitization options
        # print("  Standard assignment failed, trying alternative approach...")
        try:
            # Generate 2D coordinates for template to help with matching
            AllChem.Compute2DCoords(template_no_h)
            mol_with_bonds = AllChem.AssignBondOrdersFromTemplate(template_no_h, reordered)
        except Exception as e:
            raise ValueError(f"Could not assign bonds: {e}\n"
                           "The PDB structure may not match the SMILES, or atoms may be in incompatible order")
    
    # Add hydrogens
    if add_hydrogens:
        # print("Adding hydrogens...")
        mol_with_bonds = Chem.AddHs(mol_with_bonds, addCoords=True)
        # print(f"  Final molecule has {mol_with_bonds.GetNumAtoms()} atoms")
    
    # Write to SDF
    # print(f"Writing to {out_sdf}...")
    writer = Chem.SDWriter(out_sdf)
    writer.write(mol_with_bonds)
    writer.close()
    # print("Done!")
    
    return mol_with_bonds


def extract_ligand_id(cif_filename: str) -> str:
    """
    cif_filename format: <protein_id>_<ligand_id>_model_0.cif
    where protein_id is a string that may contain underscores
    and ligand_id is an integer
    """
    m = re.match(r"^(.+?)_(\d+)_model_0\.(cif|mmcif)$", cif_filename)
    if not m:
        raise ValueError(f"unexpected cif filename format: {cif_filename}")
    return m.group(2)


# ---------- file-level processing (one CIF) ----------
def process_one_file(
        method: str, 
        cif_path: Path, 
        outdir: Path, 
        template_dir: Optional[Path]=None, 
        smiles_lookup: Optional[dict[str, str]]=None
) -> Optional[str]:
    """
    Process a single CIF file: extract protein, ligand(s),
    write protein PDB and ligand SDF (via RDKit hydrogens).
    """
    # first check if protein/ligand already exist in outdir
    base = cif_path.stem
    out_prot = outdir / f"{base}_protein.pdb"
    out_lig = outdir / f"{base}_ligand.sdf"
    if out_prot.exists() and out_lig.exists():
        # print("Skipping", cif_path.name, "(already processed)")
        return

    print("Processing", cif_path.name)

    # create temp dir for this file
    # each cif file gets its own temp dir to avoid conflicts
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)

        prot_pdb, ligand_pdb = pymol_extract(cif_path, tmpdir)

        ligand_id = extract_ligand_id(cif_path.name)
        if method == "smiles":
            assert smiles_lookup is not None, "smiles_lookup must be provided for smiles method"
            smiles_str = smiles_lookup.get(ligand_id, None)
            if smiles_str is None:
                print(f"missing SMILES for ligand ID {ligand_id}", file=sys.stderr)
                return cif_path.name
            if ligand_pdb:
                out_lig = outdir / f"{base}_ligand.sdf"
                try:
                    assign_bonds_from_smiles_robust(smiles_str, ligand_pdb.as_posix(), out_lig.as_posix())
                except Exception as e:
                    print("Error processing", base, ":", e, file=sys.stderr)
                    return cif_path.name
        elif method == "pymol":
            assert template_dir is not None, "template_dir must be provided for pymol method"
            # check template exists
            template_file_path = template_dir / f"{ligand_id}.sdf"
            if not template_file_path.exists():
                print(f"missing template SDF for ligand ID {ligand_id}: {template_file_path}", file=sys.stderr)
                return cif_path.name
            # Convert ligand PDB -> SDF with RDKit (add H)
            # Do this step first because it may fail; upon failure, do not write protein PDB
            if ligand_pdb:
                out_lig = outdir / f"{base}_ligand.sdf"
                try:
                    rdkit_add_h_and_write_sdf(template_file_path.as_posix(), ligand_pdb.as_posix(), out_lig.as_posix())
                except Exception as e:
                    print("Error processing", base, ":", e, file=sys.stderr)
                    return cif_path.name
        else:
            print(f"unknown method: {method}", file=sys.stderr)
            return cif_path.name
            
        # Move protein PDB if present
        if prot_pdb:
            out_prot = outdir / f"{base}_protein.pdb"
            # copy file to outdir
            shutil.copy(prot_pdb, out_prot)
            
    return None  # success


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Split CIF -> protein PDB, ligand/pocket SDF using PyMOL + RDKit")
    ap.add_argument("--method", "-m", type=str, required=True, choices=["pymol", "smiles"], default="smiles", help="Method to use for processing")
    ap.add_argument("--input_dir", "-i", type=Path, required=True, help="Directory with .cif/.mmcif files")
    # only required if using SMILES-based method
    ap.add_argument("--smiles_file", "-s", type=Path, required=False, help="CSV file with drug IDs (column name: id) and their SMILES strings (column name: smiles) for smiles mode")
    ap.add_argument("--output_dir", "-o", type=Path, required=True, help="Directory to write output files")
    ap.add_argument("--template_dir", "-t", type=Path, required=False, default=None, help="Directory with template SDF files (named <ligand_id>.sdf) for pymol mode")
    ap.add_argument('--njobs', '-n', type=int, required=True, default=-1)
    args = ap.parse_args()

    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # find CIF/mmcif files
    input_dir: Path = args.input_dir
    files = sorted(list(input_dir.glob("*.cif")) + list(input_dir.glob("*.mmcif")))
    if not files:
        print("No CIF/mmcif files found in", input_dir)
        return

    if args.method == "smiles":
        if args.smiles_file:
            smiles_df = pd.read_csv(args.smiles_file)
            smiles_lookup = dict(zip(smiles_df['id'].astype(str), smiles_df['smiles'].astype(str)))
        else:
            print("Error: --smiles_file is required when using --method smiles", file=sys.stderr)
            return
    else:
        smiles_lookup = None  # not used for pymol method

    if args.method == "pymol" and not args.template_dir:
        print("Error: --template_dir is required when using --method pymol", file=sys.stderr)
        return

    # process sequentially here; for multiprocessing, call process_one_file in pool workers
    results = Parallel(n_jobs=args.njobs)(delayed(process_one_file)(args.method, cif_path, outdir, args.template_dir, smiles_lookup) for cif_path in files)
    with open("failed_files.txt", "w") as f:
        for res in results:
            if res is not None:
                f.write(res + "\n")
    # for cif_path in files:
    #     print("Processing", cif_path.name)
    #     process_one_file(cif_path, outdir, args.template_dir)

if __name__ == "__main__":
    main()
