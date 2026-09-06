"""Bemis-Murcko scaffold splitting and scaffold-grouped cross-validation.

Grouping by generic (graph) Bemis-Murcko scaffold and keeping whole scaffold
groups on one side of every split forces the model to generalize to unseen
chemical series rather than memorizing active cores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def bemis_murcko_scaffold(smiles: str, generic: bool = True) -> str:
    """Return the (optionally generic) Bemis-Murcko scaffold SMILES.

    Falls back to the input SMILES when a scaffold cannot be derived (e.g.
    acyclic molecules), so every ligand still receives a group key.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        smi = Chem.MolToSmiles(scaffold)
        return smi if smi else smiles
    except Exception:
        return smiles


def assign_scaffolds(manifest: pd.DataFrame) -> pd.Series:
    """Map each ligand to its generic Bemis-Murcko scaffold key."""
    return manifest["smiles"].map(lambda s: bemis_murcko_scaffold(s, generic=True))


def scaffold_split(
    manifest: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Greedy scaffold holdout: whole scaffold groups go entirely to train or test.

    Scaffold groups are shuffled and accumulated into the test set until it
    reaches ``test_fraction`` of the ligands. Returns
    ``(train_ids, test_ids, assignment_df)``.
    """
    scaf = assign_scaffolds(manifest)
    df = manifest[["ligand_id", "label"]].copy()
    df["scaffold"] = scaf.to_numpy()

    groups = df.groupby("scaffold")["ligand_id"].apply(list)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    grouped = list(groups.items())

    n_total = len(df)
    n_test_target = int(round(test_fraction * n_total))
    test_ids: list[str] = []
    for idx in order:
        _scaf, ids = grouped[idx]
        if len(test_ids) < n_test_target:
            test_ids.extend(ids)
    test_set = set(test_ids)
    train_ids = [i for i in df["ligand_id"] if i not in test_set]

    df["split"] = np.where(df["ligand_id"].isin(test_set), "test", "train")
    return train_ids, test_ids, df


def scaffold_kfold(
    manifest: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Assign scaffold-grouped, label-stratified CV folds.

    Uses ``StratifiedGroupKFold`` (groups = scaffold) so each molecule is
    predicted out-of-fold by a model that never saw its scaffold, while the
    active/decoy ratio is preserved across folds. Returns an assignment
    DataFrame with ``ligand_id``, ``label``, ``scaffold``, ``fold``.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    scaf = assign_scaffolds(manifest).to_numpy()
    y = manifest["label"].to_numpy()
    ids = manifest["ligand_id"].to_numpy()

    n_groups = len(set(scaf))
    n_folds = int(min(n_folds, n_groups))
    sgkf = StratifiedGroupKFold(n_splits=max(2, n_folds), shuffle=True, random_state=seed)

    fold = np.full(len(y), -1, dtype=int)
    for k, (_train, test) in enumerate(sgkf.split(np.zeros_like(y), y, groups=scaf)):
        fold[test] = k
    return pd.DataFrame({"ligand_id": ids, "label": y, "scaffold": scaf, "fold": fold})
