"""Interpretable rescoring classifiers over binary PLIF bitvectors.

Pipeline = min-frequency bit filter -> zero-variance filter -> classifier.
Two interpretable options: LightGBM with balanced class weights (captures
non-linear bit combinations) and L1-regularized logistic regression (sparse,
directly signed coefficients). Feature-name bookkeeping is preserved so the
surviving bits can be mapped back to residue-interaction labels.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline


class MinFrequencyThreshold(BaseEstimator, TransformerMixin):
    """Drop binary columns that fire in fewer than ``min_count`` training rows.

    Removes rare bits that are pure noise on small datasets, on both the
    low-frequency and (implicitly) the near-constant high-frequency ends when
    combined with ``VarianceThreshold``.
    """

    def __init__(self, min_count: int = 3) -> None:
        self.min_count = min_count

    def fit(self, X, y=None) -> "MinFrequencyThreshold":
        Xa = np.asarray(X)
        counts = Xa.sum(axis=0)
        self.support_ = counts >= self.min_count
        if not self.support_.any():  # never drop everything
            self.support_ = np.ones(Xa.shape[1], dtype=bool)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self) -> np.ndarray:
        return self.support_


def build_pipeline(model_type: str = "lightgbm", min_count: int = 3,
                   variance_threshold: float = 0.0, seed: int = 42) -> Pipeline:
    """Construct the feature-selection + classifier pipeline."""
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, class_weight="balanced",
            random_state=seed, n_jobs=1, verbosity=-1,
        )
    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5,
            class_weight="balanced", max_iter=2000, random_state=seed,
        )
    else:
        raise ValueError(f"unknown model_type: {model_type!r}")

    return Pipeline([
        ("freq", MinFrequencyThreshold(min_count=min_count)),
        ("var", VarianceThreshold(threshold=variance_threshold)),
        ("clf", clf),
    ])


def selected_feature_names(pipe: Pipeline, input_names: Sequence[str]) -> list[str]:
    """Names of the bits surviving both selection steps, in model order."""
    names = np.asarray(input_names)
    names = names[pipe.named_steps["freq"].get_support()]
    names = names[pipe.named_steps["var"].get_support()]
    return names.tolist()


def feature_importances(pipe: Pipeline, input_names: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame of surviving bits with signed/gain importance.

    LightGBM -> gain importance (unsigned). LogReg -> raw coefficient (signed)
    plus its absolute value for ranking.
    """
    names = selected_feature_names(pipe, input_names)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "booster_"):
        gain = clf.booster_.feature_importance(importance_type="gain")
        out = pd.DataFrame({"feature": names, "importance": gain})
        out["signed"] = np.nan
    else:
        coef = np.ravel(clf.coef_)
        out = pd.DataFrame({"feature": names, "importance": np.abs(coef), "signed": coef})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
