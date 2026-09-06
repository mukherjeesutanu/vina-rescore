"""Virtual-screening metrics: ROC-AUC, BEDROC(alpha), and enrichment factors.

All functions take ``y_true`` (1=active, 0=decoy) and ``y_score`` where a
*higher* score means *more likely active*. For raw Vina affinity (more
negative = better) pass ``-affinity``.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def _ranked_labels(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Labels ordered by descending score (stable; ties keep input order)."""
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    return np.asarray(y_true, dtype=int)[order]


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) identity, with tie handling."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    # average ranks (1-indexed) with ties shared
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_pos = ranks[np.isin(order, np.where(y == 1)[0])].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def enrichment_factor(y_true: Sequence[int], y_score: Sequence[float],
                      fraction: float) -> float:
    """EF at the top ``fraction`` (e.g. 0.01 for EF1%)."""
    y = _ranked_labels(np.asarray(y_true), np.asarray(y_score))
    n = len(y)
    n_actives = int(y.sum())
    if n_actives == 0:
        return float("nan")
    n_top = max(1, int(round(fraction * n)))
    hits_top = int(y[:n_top].sum())
    return (hits_top / n_top) / (n_actives / n)


def max_enrichment_factor(y_true: Sequence[int], fraction: float) -> float:
    """Theoretical ceiling for EF at ``fraction`` given the active ratio."""
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    n_actives = int(y.sum())
    if n_actives == 0:
        return float("nan")
    n_top = max(1, int(round(fraction * n)))
    return (min(n_top, n_actives) / n_top) / (n_actives / n)


def bedroc(y_true: Sequence[int], y_score: Sequence[float], alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly, J. Chem. Inf. Model. 2007), early-recognition metric.

    Returns a value in [0, 1]; 0.5 is random for a balanced set, higher means
    actives are concentrated near the top of the ranking.
    """
    y = _ranked_labels(np.asarray(y_true), np.asarray(y_score))
    n = len(y)
    n_actives = int(y.sum())
    if n_actives == 0 or n_actives == n:
        return float("nan")

    ranks = np.where(y == 1)[0] + 1  # 1-indexed ranks of actives
    ra = n_actives / n
    # Robust Initial Enhancement, normalized by the random expectation
    sum_exp = np.sum(np.exp(-alpha * ranks / n))
    rie_random = (ra * (1.0 - math.exp(-alpha))) / (math.exp(alpha / n) - 1.0)
    rie = sum_exp / rie_random
    # map RIE onto [0, 1]
    factor = (ra * math.sinh(alpha / 2.0)) / (
        math.cosh(alpha / 2.0) - math.cosh(alpha / 2.0 - alpha * ra)
    )
    const = 1.0 / (1.0 - math.exp(alpha * (1.0 - ra)))
    return float(rie * factor + const)


def evaluate(y_true: Sequence[int], y_score: Sequence[float],
             alpha: float = 20.0,
             ef_fractions: Sequence[float] = (0.01, 0.05)) -> dict[str, float]:
    """Compute the full metric panel for one score vector."""
    out: dict[str, float] = {
        "ROC_AUC": roc_auc(y_true, y_score),
        f"BEDROC_a{int(alpha)}": bedroc(y_true, y_score, alpha),
    }
    for f in ef_fractions:
        out[f"EF_{f*100:g}%"] = enrichment_factor(y_true, y_score, f)
    return out


def _stratified_boot_index(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample indices with replacement *within* each class (preserves the
    active/decoy ratio so EF/BEDROC stay well defined)."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)])


def bootstrap_cis(y_true: Sequence[int], y_score: Sequence[float],
                  alpha: float = 20.0,
                  ef_fractions: Sequence[float] = (0.01, 0.05),
                  n_boot: int = 2000, seed: int = 0,
                  ci: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals for each metric in
    :func:`evaluate`, using class-stratified resampling of ligands."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    keys = list(evaluate(y, s, alpha, ef_fractions).keys())
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        bi = _stratified_boot_index(y, rng)
        yb = y[bi]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        m = evaluate(yb, s[bi], alpha, ef_fractions)
        for k in keys:
            draws[k].append(m[k])
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    return {k: (float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q)))
            for k, v in draws.items()}


def paired_bootstrap(y_true: Sequence[int], score_a: Sequence[float],
                     score_b: Sequence[float], alpha: float = 20.0,
                     ef_fractions: Sequence[float] = (0.01, 0.05),
                     n_boot: int = 2000, seed: int = 0,
                     ci: float = 0.95) -> dict[str, dict[str, float]]:
    """Paired bootstrap of the metric *difference* A-B (same resample applied to
    both scorers). Returns per-metric ``{delta, lo, hi, p_gt}`` where ``p_gt``
    is the fraction of replicates with A>B (a one-sided bootstrap p-proxy)."""
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    keys = list(evaluate(y, a, alpha, ef_fractions).keys())
    diffs: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        bi = _stratified_boot_index(y, rng)
        yb = y[bi]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ma = evaluate(yb, a[bi], alpha, ef_fractions)
        mb = evaluate(yb, b[bi], alpha, ef_fractions)
        for k in keys:
            diffs[k].append(ma[k] - mb[k])
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out: dict[str, dict[str, float]] = {}
    for k, v in diffs.items():
        arr = np.asarray(v)
        out[k] = {"delta": float(arr.mean()),
                  "lo": float(np.percentile(arr, lo_q)),
                  "hi": float(np.percentile(arr, hi_q)),
                  "p_gt": float((arr > 0).mean())}
    return out
