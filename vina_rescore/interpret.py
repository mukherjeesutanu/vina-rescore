"""Biophysical interpretation of the most predictive PLIF bits.

Maps ``RESID:INTERACTION`` bits to CDK2 ATP-pocket roles (hinge, catalytic
lysine, gatekeeper, DFG, glycine-rich loop) so the top features can be judged
against known mechanistic binding determinants.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# CDK2 (human, UniProt P24941) ATP-pocket residue roles for interpretation.
CDK2_RESIDUE_ROLES: dict[int, str] = {
    10: "P-loop (Ile10)",
    11: "glycine-rich loop", 12: "glycine-rich loop", 13: "glycine-rich loop",
    14: "glycine-rich loop", 15: "glycine-rich loop", 16: "glycine-rich loop",
    18: "P-loop (Val18)",
    31: "Ala31 (pocket floor)",
    33: "catalytic Lys33 (beta3)",
    51: "Glu51 (alphaC helix)",
    64: "Val64",
    80: "gatekeeper Phe80",
    81: "hinge Glu81",
    82: "hinge Phe82",
    83: "hinge Leu83",
    84: "hinge His84",
    85: "Gln85",
    86: "Asp86",
    89: "Lys89 (solvent front)",
    129: "Leu134 region",
    131: "Asp127/back pocket",
    144: "Ala144",
    145: "DFG Asp145",
}
HINGE_RESIDUES = {81, 82, 83, 84}

_RES_RE = re.compile(r"^([A-Z]{2,3})(\d+)")


def parse_bit(bit: str) -> tuple[str, Optional[int], str]:
    """Split a ``RESID:INTERACTION`` bit into ``(resname, resnum, interaction)``."""
    res, _, interaction = bit.partition(":")
    m = _RES_RE.match(res)
    if not m:
        return res, None, interaction
    return m.group(1), int(m.group(2)), interaction


def annotate_role(bit: str) -> str:
    """Human-readable CDK2 role for the residue in a bit ('' if unknown)."""
    _res, num, _inter = parse_bit(bit)
    if num is None:
        return ""
    role = CDK2_RESIDUE_ROLES.get(num, "")
    if num in HINGE_RESIDUES and role:
        role += " [HINGE]"
    return role


def top_features(importance: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Return the top-``k`` bits with parsed residue/interaction and CDK2 role."""
    top = importance.head(k).copy()
    parsed = top["feature"].map(parse_bit)
    top["residue"] = [f"{r}{n}" if n is not None else r for r, n, _ in parsed]
    top["interaction"] = [i for _, _, i in parsed]
    top["cdk2_role"] = top["feature"].map(annotate_role)
    return top.reset_index(drop=True)


def add_bit_enrichment(top: pd.DataFrame, plif: pd.DataFrame) -> pd.DataFrame:
    """Annotate top features with their per-class firing frequency.

    For each bit, ``active_freq`` / ``decoy_freq`` are the fractions of active
    and decoy poses in which the interaction is present, and ``enrich_ratio`` is
    their quotient. This separates *importance* (how much the model leans on a
    bit) from *discriminative power* (how differently it fires between classes) —
    a high-importance bit that fires similarly in both classes is a weak
    mechanistic signal.
    """
    out = top.copy()
    act = plif[plif["label"] == 1]
    dec = plif[plif["label"] == 0]
    af, dfr, ratio = [], [], []
    for bit in out["feature"]:
        if bit in plif.columns:
            a = float(act[bit].mean())
            d = float(dec[bit].mean())
            af.append(a); dfr.append(d)
            ratio.append(a / d if d > 0 else float("inf"))
        else:
            af.append(float("nan")); dfr.append(float("nan")); ratio.append(float("nan"))
    out["active_freq"] = af
    out["decoy_freq"] = dfr
    out["enrich_ratio"] = ratio
    return out
