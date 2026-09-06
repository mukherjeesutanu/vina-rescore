"""Central configuration for the vina_rescore pipeline.

All tunable knobs live here so a run is fully described by a single ``Config``
instance. Defaults are sized for a laptop (Core i5 / 16 GB RAM): a small DUD-E
CDK2 subset docked at modest Vina exhaustiveness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence
import json


# prolif interaction types used to build the binary fingerprint.
# Covers the mechanistically relevant contacts for an ATP-site kinase:
# hydrogen bonds (both directions), hydrophobic packing, pi-stacking and
# cation-pi (both directions).
DEFAULT_INTERACTIONS: tuple[str, ...] = (
    "HBDonor",
    "HBAcceptor",
    "Hydrophobic",
    "PiStacking",
    "CationPi",
    "PiCation",
)


@dataclass
class Config:
    """Typed configuration for a full rescoring run."""

    # --- filesystem layout ---
    root: Path = Path(".")
    raw_dir: Path = Path("data/raw/cdk2")
    work_dir: Path = Path("data")
    out_dir: Path = Path("outputs")
    fig_dir: Path = Path("outputs/figures")

    # --- dataset (DUD-E CDK2) subsampling ---
    n_actives: int = 50
    n_decoys: int = 300
    random_seed: int = 42

    # --- docking box (Angstrom); center may be auto-derived from crystal ligand ---
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] = (22.5, 22.5, 22.5)

    # --- ligand preparation ---
    ph: float = 7.4
    embed_seed: int = 42

    # --- Vina docking ---
    exhaustiveness: int = 8
    n_poses: int = 5
    n_workers: int = 8          # parallel docking processes
    cpu_per_worker: int = 1     # Vina threads per process
    dock_timeout_s: int = 300   # per-ligand wall-clock guard

    # --- PLIF extraction ---
    interactions: Sequence[str] = DEFAULT_INTERACTIONS
    pocket_cutoff: float = 6.0  # Angstrom (prolif vicinity_cutoff)

    # --- feature selection / model ---
    variance_threshold: float = 0.0     # drop zero-variance columns
    min_bit_frequency: int = 3          # a bit must fire in >= this many poses
    model_type: str = "lightgbm"        # "lightgbm" | "logreg"

    # --- evaluation ---
    cv_folds: int = 5
    test_fraction: float = 0.25         # single scaffold holdout size
    bedroc_alpha: float = 20.0
    ef_fractions: tuple[float, ...] = (0.01, 0.05)
    top_k_features: int = 5

    def resolve(self) -> "Config":
        """Return a copy with all paths made absolute relative to ``root``."""
        r = self.root.resolve()
        self.raw_dir = (r / self.raw_dir).resolve() if not self.raw_dir.is_absolute() else self.raw_dir
        self.work_dir = (r / self.work_dir).resolve() if not self.work_dir.is_absolute() else self.work_dir
        self.out_dir = (r / self.out_dir).resolve() if not self.out_dir.is_absolute() else self.out_dir
        self.fig_dir = (r / self.fig_dir).resolve() if not self.fig_dir.is_absolute() else self.fig_dir
        for d in (self.work_dir, self.out_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
        Path(path).write_text(json.dumps(d, indent=2, default=list))
