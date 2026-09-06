"""Publication-quality figures for the rescoring benchmark (self-contained).

Follows scientific-figure correctness conventions: limited hues with a single
focal series (the PLIF rescorer) against a neutral Vina baseline, direct
labels, sentence titles, and an explicit direction-of-goodness cue.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOCAL = "#1f6f8b"     # PLIF rescorer
BASELINE = "#9aa0a6"  # raw Vina
RANDOM = "#c8102e"    # random reference (alarm hue, reference only)


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",
    })


def _cumulative_recall(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="stable")
    y = y_true[order]
    frac_screened = np.arange(1, len(y) + 1) / len(y)
    recall = np.cumsum(y) / max(int(y.sum()), 1)
    return frac_screened, recall


def enrichment_curve(oof: pd.DataFrame, out: Path) -> Path:
    """Cumulative actives recovered vs fraction of the library screened."""
    setup_style()
    y = oof["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for score, color, name in [
        (oof["rescore"].to_numpy(), FOCAL, "PLIF rescorer"),
        (oof["vina_score"].to_numpy(), BASELINE, "raw Vina"),
    ]:
        fs, rc = _cumulative_recall(y, score)
        ax.plot(fs * 100, rc * 100, color=color, label=name,
                lw=2.2 if name.startswith("PLIF") else 1.6)
    ax.plot([0, 100], [0, 100], color=RANDOM, ls=":", lw=1.2, label="random")
    for xf in (1, 5):
        ax.axvline(xf, color="0.8", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax.set_xlabel("library screened (%)")
    ax.set_ylabel("actives recovered (%)")
    ax.set_title("PLIF rescoring improves early enrichment (top ~5%)", loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.5, 0.06, "up / left = better", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, color="0.4", style="italic")
    fig.savefig(out)
    plt.close(fig)
    return out


def roc_curves(oof: pd.DataFrame, out: Path) -> Path:
    """ROC curves for the rescorer and raw Vina."""
    from sklearn.metrics import roc_curve, roc_auc_score
    setup_style()
    y = oof["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for score, color, name in [
        (oof["rescore"].to_numpy(), FOCAL, "PLIF rescorer"),
        (oof["vina_score"].to_numpy(), BASELINE, "raw Vina"),
    ]:
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.2f})",
                lw=2.2 if name.startswith("PLIF") else 1.6)
    ax.plot([0, 1], [0, 1], color=RANDOM, ls=":", lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC: pooled scaffold-CV predictions", loc="left")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out)
    plt.close(fig)
    return out


def metric_bars(table: pd.DataFrame, out: Path, evaluation: str,
                ef_ceiling: dict[str, float] | None = None) -> Path:
    """Grouped bars comparing the metric panel (rescorer vs Vina)."""
    setup_style()
    sub = table[table["evaluation"] == evaluation].set_index("scorer")
    metrics = [c for c in table.columns if c not in ("evaluation", "scorer")]
    bounded = [m for m in metrics if m.startswith(("ROC", "BEDROC"))]
    ef = [m for m in metrics if m.startswith("EF")]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2),
                             gridspec_kw={"width_ratios": [len(bounded), len(ef)]})
    order = ["PLIF rescorer", "raw Vina"]
    colors = [FOCAL, BASELINE]
    for ax, group, title in [(axes[0], bounded, "ranking quality (0-1)"),
                             (axes[1], ef, "early enrichment (x over random)")]:
        x = np.arange(len(group)); w = 0.38
        for i, sc in enumerate(order):
            vals = [sub.loc[sc, m] for m in group]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colors[i],
                          label=sc, edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(group, rotation=0)
        ax.set_title(title, loc="left", fontsize=8)
        ax.margins(y=0.18)
        if group is ef and ef_ceiling:
            for xi, m in zip(x, group):
                c = ef_ceiling.get(m)
                if c:
                    ax.hlines(c, xi - 0.5, xi + 0.5, color="0.5", ls="--", lw=0.8)
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].text(0.98, 0.98, "dashed = EF ceiling", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=6.5, color="0.4", style="italic")
    fig.suptitle(f"PLIF rescorer vs raw Vina — {evaluation}", x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    return out


def feature_bars(top: pd.DataFrame, out: Path) -> Path:
    """Horizontal bar of the top predictive PLIF bits with CDK2 roles."""
    setup_style()
    t = top.iloc[::-1]  # largest on top
    labels = []
    for r, i, role in zip(t["residue"], t["interaction"], t["cdk2_role"]):
        base = f"{r} · {i}"
        labels.append(f"{base}\n{role}" if role else base)
    fig, ax = plt.subplots(figsize=(5.8, 0.62 * len(t) + 1.4))
    signed = t["signed"].to_numpy() if "signed" in t and t["signed"].notna().any() else None
    vals = t["importance"].to_numpy()
    colors = [FOCAL if (signed is None or s >= 0) else RANDOM
              for s in (signed if signed is not None else vals)]
    ax.barh(range(len(t)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(labels)
    ax.set_xlabel("importance (gain)" if signed is None else "|coefficient|")
    ax.set_title("Top predictive interactions map to the CDK2 ATP pocket", loc="left")
    ax.margins(x=0.10)
    if signed is not None:
        ax.text(0.98, 0.02, "blue = favors active · red = favors decoy",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.4")
    fig.savefig(out)
    plt.close(fig)


def size_decorrelation(df: pd.DataFrame, out: Path,
                       rho_rescore: float, rho_vina: float) -> Path:
    """Score percentile-rank vs ligand heavy-atom count for both scorers.

    A size-biased scorer trends upward (larger ligands ranked better) — a known
    DUD-E artifact. ``df`` needs columns ``heavy``, ``rescore``, ``vina_score``,
    ``label``; ``rho_*`` are the precomputed Spearman coefficients.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    n = len(df)
    heavy = df["heavy"].to_numpy()
    panels = [
        (axes[0], df["rescore"].to_numpy(), FOCAL, "PLIF rescorer", rho_rescore),
        (axes[1], df["vina_score"].to_numpy(), BASELINE, "raw Vina", rho_vina),
    ]
    for ax, score, color, name, rho in panels:
        rank = (np.argsort(np.argsort(score)) / max(n - 1, 1)) * 100  # percentile rank
        act = df["label"].to_numpy() == 1
        ax.scatter(heavy[~act], rank[~act], s=9, c="0.75", alpha=0.6,
                   linewidths=0, label="decoy")
        ax.scatter(heavy[act], rank[act], s=14, c=color, alpha=0.9,
                   linewidths=0, label="active")
        # linear trend for the eye
        b, a = np.polyfit(heavy, rank, 1)
        xs = np.array([heavy.min(), heavy.max()])
        ax.plot(xs, a + b * xs, color=color, lw=1.6)
        ax.set_title(f"{name}\nSpearman \u03c1 = {rho:+.2f}", loc="left")
        ax.set_xlabel("ligand heavy atoms")
    axes[0].set_ylabel("score percentile rank")
    axes[0].set_ylim(-2, 102)
    axes[1].legend(loc="lower right", frameon=False, markerscale=1.3)
    fig.suptitle("PLIF ranking is size-decorrelated; raw Vina favors larger ligands",
                 x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    return out
    return out
