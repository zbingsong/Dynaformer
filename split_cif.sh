#!/usr/bin/env bash
set -euo pipefail

# Usage: cif_to_pdb_sdf_bash.sh /path/to/input_dir /path/to/output_dir [distance]
# Example: cif_to_pdb_sdf_bash.sh ./cifs ./out 5.0

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
DISTANCE="${3:-5.0}"

# were we sourced? (return succeeds only in a sourced context)
if (return 0 2>/dev/null); then
  sourced=1
else
  sourced=0
fi

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 /path/to/input_dir /path/to/output_dir [distance_angstrom]"
  if [[ $sourced -eq 1 ]]; then
    return 1    # only works when sourced
  else
    exit 1      # safe when executed
  fi
fi

# Check tools
command -v pymol >/dev/null 2>&1 || { echo "pymol not found in PATH. Please install or give full path."; exit 1; }
command -v obabel >/dev/null 2>&1 || { echo "obabel not found in PATH. Please install OpenBabel."; exit 1; }

mkdir -p -- "$OUTPUT_DIR"

# enable nullglob so the loop doesn't expand to literal patterns
shopt -s nullglob

tmpd="$(mktemp -d)"
script_pml="$tmpd/script.pml"
ligand_raw="$tmpd/ligand_raw.pdb"
protein_raw="$tmpd/protein_raw.pdb"
pocket_raw="$tmpd/pocket_raw.pdb"

for file in "$INPUT_DIR"/*.{cif,mmcif}; do
  [[ -e "$file" ]] || continue
  base="$(basename "$file")"
  base="${base%.*}"        # strip single extension (.cif/.mmcif)
  echo "== Processing: $base =="

  # create PyMOL .pml (we use alter to force resn='LIG')
  cat > "$script_pml" <<-PYML
load $file, myobj
select ligands, myobj and not polymer and not resn HOH
alter ligands, resn='LIG'
save $ligand_raw, ligands
select protein_sel, myobj and polymer and not resn HOH
save $protein_raw, protein_sel
select pocket_sel, byres (myobj and polymer and not resn HOH within $DISTANCE of ligands)
save $pocket_raw, pocket_sel
quit
PYML

  # run PyMOL in batch mode (quiet)
  if ! pymol -cq "$script_pml"; then
    echo "PyMOL failed for $file (check file or PyMOL installation)."
    continue
  fi

  # 1) protein: tidy + element (leave as final protein PDB)
  mv "$protein_raw" "$OUTPUT_DIR/${base}_protein.pdb"
  echo "  -> protein PDB: ${base}_protein.pdb"

  # 2) ligand -> tidy/element -> obabel -> SDF
  if obabel -ipdb "$ligand_raw" -osdf -O "$OUTPUT_DIR/${base}_ligand.sdf" -x3v -h --partialcharge eem; then
    echo "  -> ligand SDF: ${base}_ligand.sdf (with -h)"
  else
    echo "  [ERROR] obabel failed to convert ligand for $base"
    continue
  fi

done

# cleanup
rm -rf "$tmpd"

echo "All done."
