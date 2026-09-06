"""Receptor and ligand preparation.

Receptor: DUD-E ``receptor.pdb`` -> rigid PDBQT (for Vina) and a fully
protonated PDB (for prolif), both via OpenBabel at the configured pH.

Ligand: SMILES -> RDKit 3D embed (ETKDGv3) -> MMFF minimize -> Meeko PDBQT.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

from .config import Config


class PreparationError(RuntimeError):
    """Raised when a molecule cannot be prepared for docking."""


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreparationError(f"command failed ({' '.join(cmd)}):\n{proc.stderr[-500:]}")


def prepare_receptor(cfg: Config) -> tuple[Path, Path]:
    """Produce ``receptor.pdbqt`` (rigid, for Vina) and ``receptor_H.pdb``
    (protonated, for prolif). Returns ``(pdbqt_path, protonated_pdb_path)``.
    """
    src = cfg.raw_dir / "receptor.pdb"
    if not src.exists():
        raise FileNotFoundError(src)
    pdbqt = cfg.work_dir / "receptor.pdbqt"
    pdb_h = cfg.work_dir / "receptor_H.pdb"

    # rigid receptor PDBQT with Gasteiger charges + polar H at pH
    _run(["obabel", str(src), "-O", str(pdbqt), "-xr", "-p", str(cfg.ph)])
    # fully protonated PDB for interaction fingerprinting
    _run(["obabel", str(src), "-O", str(pdb_h), "-p", str(cfg.ph)])
    if not pdbqt.exists() or pdbqt.stat().st_size == 0:
        raise PreparationError("receptor PDBQT was not written")
    return pdbqt, pdb_h


def embed_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    """SMILES -> protonated, 3D-embedded, MMFF-minimized RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PreparationError(f"unparseable SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # retry with random coordinates as a fallback
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise PreparationError(f"3D embedding failed: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # geometry is embedded; minimization is best-effort
    return mol


def prepare_ligand_pdbqt(smiles: str, seed: int = 42) -> str:
    """SMILES -> Meeko PDBQT string ready for Vina. Raises on failure."""
    mol = embed_3d(smiles, seed=seed)
    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise PreparationError(f"Meeko produced no setup: {smiles}")
    pdbqt, ok, msg = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise PreparationError(f"Meeko PDBQT export failed: {msg}")
    return pdbqt
