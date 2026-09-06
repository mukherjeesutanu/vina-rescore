"""Protein-Ligand Interaction Fingerprint (PLIF) extraction with prolif.

The protonated receptor is loaded once via MDAnalysis; each docked pose is
loaded (with explicit hydrogens) and fingerprinted against pocket residues
within ``cfg.pocket_cutoff`` (default 6.0 A). The result is a binary matrix
whose columns are ``RESID:INTERACTION`` bits.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# Nonstandard 3-letter residue codes emitted by some receptor-prep pipelines
# (DUD-E, AMBER protonation states) mapped to canonical amino-acid codes.
# These affect labels only; prolif detects interactions from atoms/coordinates,
# not from the residue name.
_RESNAME_CANON: dict[str, str] = {
    "LEV": "LEU",  # DUD-E CDK2 labels the hinge Leu83 as "LEV"
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",  # His protonation states
    "GLH": "GLU", "ASH": "ASP",               # protonated acidic residues
    "LYN": "LYS", "CYX": "CYS", "CYM": "CYS",  # neutral Lys / Cys variants
}


def _canonical_residue(residue: str) -> str:
    """Normalize a prolif residue label (e.g. ``LEV83`` / ``LEV83.A``) so the
    3-letter code is a canonical amino acid, preserving the number and chain."""
    resid = str(residue)
    i = 0
    while i < len(resid) and resid[i].isalpha():
        i += 1
    code, rest = resid[:i], resid[i:]
    return _RESNAME_CANON.get(code.upper(), code) + rest


def _load_protein(receptor_h_pdb: Path):
    import MDAnalysis as mda
    import prolif as plf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(receptor_h_pdb))
        return plf.Molecule.from_mda(u, NoImplicit=False)


def _iter_pose_molecules(docked_sdf: Path):
    """Yield ``(ligand_id, prolif.Molecule)`` for each docked pose, with Hs."""
    from rdkit import Chem
    import prolif as plf

    supplier = Chem.SDMolSupplier(str(docked_sdf), removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        mol_h = Chem.AddHs(mol, addCoords=True)
        yield name, plf.Molecule.from_rdkit(mol_h)


def extract_plif(cfg: Config, receptor_h_pdb: Path, docked_sdf: Path,
                 manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the binary PLIF matrix for all docked poses.

    Returns a DataFrame indexed by ``ligand_id`` with a ``label`` column and
    one 0/1 column per ``RESID:INTERACTION`` bit. Also written to
    ``<out_dir>/plif_matrix.csv``.
    """
    import prolif as plf

    protein = _load_protein(receptor_h_pdb)

    ligand_ids: list[str] = []
    mols = []
    for lig_id, m in _iter_pose_molecules(docked_sdf):
        ligand_ids.append(lig_id)
        mols.append(m)
    if not mols:
        raise RuntimeError(f"no poses parsed from {docked_sdf}")

    fp = plf.Fingerprint(
        interactions=list(cfg.interactions),
        count=False,
        vicinity_cutoff=cfg.pocket_cutoff,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp.run_from_iterable(mols, protein, progress=False)
    df = fp.to_dataframe()

    # flatten (ligand, residue, interaction) -> "RESID:INTERACTION" bits
    flat = pd.DataFrame(index=range(len(mols)))
    for col in df.columns:
        residue, interaction = col[1], col[2]
        name = f"{_canonical_residue(residue)}:{interaction}"
        vals = df[col].astype(int).to_numpy()
        if name in flat.columns:
            flat[name] = np.maximum(flat[name].to_numpy(), vals)
        else:
            flat[name] = vals

    flat.insert(0, "ligand_id", ligand_ids)
    label_map = dict(zip(manifest["ligand_id"], manifest["label"]))
    flat.insert(1, "label", flat["ligand_id"].map(label_map).astype(int))
    flat = flat.sort_index(axis=1)  # deterministic bit column order
    # restore leading columns order
    lead = ["ligand_id", "label"]
    bits = [c for c in flat.columns if c not in lead]
    flat = flat[lead + bits]

    flat.to_csv(cfg.out_dir / "plif_matrix.csv", index=False)
    return flat
