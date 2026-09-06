"""DUD-E ingestion, manifest construction, and docking-box derivation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config


def _parse_ism(path: Path, label: int) -> pd.DataFrame:
    """Parse a DUD-E ``*_final.ism`` file (``SMILES id [extra_id]`` per line)."""
    rows: list[dict[str, object]] = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append(
                {
                    "raw_id": parts[1],
                    "smiles": parts[0],
                    "label": int(label),
                    "source": "active" if label == 1 else "decoy",
                    "extra_id": parts[2] if len(parts) > 2 else "",
                }
            )
    return pd.DataFrame(rows)


def build_manifest(cfg: Config) -> pd.DataFrame:
    """Subsample DUD-E actives/decoys into a reproducible ligand manifest.

    Returns a DataFrame with unique ``ligand_id``, ``smiles``, ``label``
    (1=active, 0=decoy) and ``source`` columns, written to
    ``<work_dir>/manifest.csv``.
    """
    actives = _parse_ism(cfg.raw_dir / "actives_final.ism", 1)
    decoys = _parse_ism(cfg.raw_dir / "decoys_final.ism", 0)

    act = actives.sample(n=min(cfg.n_actives, len(actives)), random_state=cfg.random_seed)
    dec = decoys.sample(n=min(cfg.n_decoys, len(decoys)), random_state=cfg.random_seed)
    man = pd.concat([act, dec], ignore_index=True)

    # unique, filesystem-safe id
    man["ligand_id"] = man["source"].str[:3] + "_" + man["raw_id"].astype(str)
    man = man.drop_duplicates(subset="ligand_id").reset_index(drop=True)
    if not man["ligand_id"].is_unique:
        raise ValueError("ligand_id collisions after subsampling")

    man = man[["ligand_id", "smiles", "label", "source", "extra_id"]]
    out = cfg.work_dir / "manifest.csv"
    man.to_csv(out, index=False)
    return man


def mol2_coordinates(path: Path) -> np.ndarray:
    """Extract ``(N, 3)`` atomic coordinates from a Tripos MOL2 file."""
    coords: list[list[float]] = []
    in_atom = False
    for line in open(path):
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>") and in_atom:
            break
        if in_atom:
            p = line.split()
            if len(p) >= 5:
                coords.append([float(p[2]), float(p[3]), float(p[4])])
    if not coords:
        raise ValueError(f"no atoms parsed from {path}")
    return np.asarray(coords, dtype=float)


def docking_box(cfg: Config) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Derive the docking-box center from the co-crystallized ligand.

    The center is the crystal-ligand centroid; the size is taken from
    ``cfg.box_size`` (default 22.5 A cube), which comfortably encloses the
    CDK2 ATP pocket. If ``cfg.box_center`` is set it overrides the centroid.
    """
    if cfg.box_center is not None:
        center = tuple(float(x) for x in cfg.box_center)
    else:
        xtal = mol2_coordinates(cfg.raw_dir / "crystal_ligand.mol2")
        center = tuple(float(x) for x in xtal.mean(axis=0))
    size = tuple(float(x) for x in cfg.box_size)
    return center, size  # type: ignore[return-value]
