"""End-to-end orchestration of the rescoring benchmark (post-docking)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metrics import evaluate, max_enrichment_factor
from .models import build_pipeline, feature_importances
from .splits import scaffold_kfold, scaffold_split


@dataclass
class BenchmarkResult:
    table: pd.DataFrame            # metrics: rescorer vs raw Vina (CV-pooled + holdout)
    oof: pd.DataFrame             # per-ligand out-of-fold scores + labels
    importance: pd.DataFrame      # feature importances (model trained on all data)
    fold_assignment: pd.DataFrame
    holdout_assignment: pd.DataFrame


def _feature_frame(plif: pd.DataFrame, scores: pd.DataFrame,
                   manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join PLIF bits with Vina scores + SMILES on successfully docked ligands."""
    docked = scores.dropna(subset=["vina_affinity"])[["ligand_id", "vina_affinity"]]
    df = (plif.merge(docked, on="ligand_id", how="inner")
              .merge(manifest[["ligand_id", "smiles"]], on="ligand_id", how="left"))
    bit_cols = [c for c in plif.columns if c not in ("ligand_id", "label")]
    return df, bit_cols


def run_benchmark(cfg: Config, plif: pd.DataFrame, scores: pd.DataFrame,
                  manifest: pd.DataFrame) -> BenchmarkResult:
    """Scaffold-aware CV benchmark of the PLIF rescorer against raw Vina.

    Primary evaluation pools out-of-fold predictions from scaffold-grouped,
    label-stratified CV (every molecule scored by a model blind to its
    scaffold), which yields full-dataset early-enrichment estimates on a small
    set. A single scaffold holdout is also reported.
    """
    df, bit_cols = _feature_frame(plif, scores, manifest)
    X = df[bit_cols].to_numpy()
    y = df["label"].to_numpy().astype(int)
    vina_score = -df["vina_affinity"].to_numpy()  # higher = better

    # ---- scaffold-grouped CV: pooled out-of-fold model probabilities ----
    folds = scaffold_kfold(df, n_folds=cfg.cv_folds, seed=cfg.random_seed)
    fold_id = folds["fold"].to_numpy()
    oof_pred = np.full(len(y), np.nan)
    for k in np.unique(fold_id):
        tr = fold_id != k
        te = fold_id == k
        pipe = build_pipeline(cfg.model_type, cfg.min_bit_frequency,
                              cfg.variance_threshold, cfg.random_seed)
        pipe.fit(X[tr], y[tr])
        oof_pred[te] = pipe.predict_proba(X[te])[:, 1]

    oof = pd.DataFrame({
        "ligand_id": df["ligand_id"].to_numpy(),
        "label": y, "fold": fold_id,
        "rescore": oof_pred, "vina_score": vina_score,
        "vina_affinity": df["vina_affinity"].to_numpy(),
    })

    ef_fracs = cfg.ef_fractions
    rows = []
    m_model = evaluate(y, oof_pred, cfg.bedroc_alpha, ef_fracs)
    m_vina = evaluate(y, vina_score, cfg.bedroc_alpha, ef_fracs)
    rows.append({"evaluation": "scaffold-CV (pooled OOF)", "scorer": "PLIF rescorer", **m_model})
    rows.append({"evaluation": "scaffold-CV (pooled OOF)", "scorer": "raw Vina", **m_vina})

    # ---- single scaffold holdout ----
    train_ids, test_ids, holdout = scaffold_split(df, cfg.test_fraction, cfg.random_seed)
    tr_mask = df["ligand_id"].isin(set(train_ids)).to_numpy()
    te_mask = df["ligand_id"].isin(set(test_ids)).to_numpy()
    pipe = build_pipeline(cfg.model_type, cfg.min_bit_frequency,
                          cfg.variance_threshold, cfg.random_seed)
    pipe.fit(X[tr_mask], y[tr_mask])
    p_test = pipe.predict_proba(X[te_mask])[:, 1]
    rows.append({"evaluation": "scaffold holdout (test)", "scorer": "PLIF rescorer",
                 **evaluate(y[te_mask], p_test, cfg.bedroc_alpha, ef_fracs)})
    rows.append({"evaluation": "scaffold holdout (test)", "scorer": "raw Vina",
                 **evaluate(y[te_mask], vina_score[te_mask], cfg.bedroc_alpha, ef_fracs)})

    table = pd.DataFrame(rows)

    # ---- interpretation model: fit on all docked data ----
    full = build_pipeline(cfg.model_type, cfg.min_bit_frequency,
                          cfg.variance_threshold, cfg.random_seed)
    full.fit(X, y)
    importance = feature_importances(full, bit_cols)

    # attach EF ceilings for context
    table.attrs["ef_ceiling"] = {f"EF_{f*100:g}%": max_enrichment_factor(y, f)
                                 for f in ef_fracs}

    table.to_csv(cfg.out_dir / "benchmark_results.csv", index=False)
    oof.to_csv(cfg.out_dir / "oof_predictions.csv", index=False)
    importance.to_csv(cfg.out_dir / "feature_importance.csv", index=False)
    return BenchmarkResult(table, oof, importance, folds, holdout)
