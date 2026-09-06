"""vina_rescore: rescore AutoDock Vina docking poses for early enrichment.

A modular pipeline that turns protein-ligand interaction fingerprints (PLIFs)
of Vina poses into an interpretable classifier whose ranking beats raw Vina
affinity on early-enrichment metrics (BEDROC, EF1%, EF5%).

Modules
-------
config      : central, typed configuration.
data        : DUD-E ingestion, manifest building, docking-box definition.
prep        : receptor and ligand preparation (RDKit + Meeko + OpenBabel).
docking     : parallel AutoDock Vina docking engine.
plif        : prolif/MDAnalysis binary interaction-fingerprint extraction.
splits      : Bemis-Murcko scaffold splitting and scaffold-grouped CV.
models      : interpretable classifiers (LightGBM / L1 logistic regression).
metrics     : ROC-AUC, BEDROC(alpha), EF at top fractions.
interpret   : top predictive residue-interaction features.
pipeline    : end-to-end orchestration.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import Config

__all__ = ["Config", "__version__"]
