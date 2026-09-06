# Referee Report — PLIF Rescoring of Vina Poses (DUD-E CDK2)

**Verdict:** Methodologically sound and honestly framed; the engineering and
evaluation design are strong. The central *scientific* claim — that PLIF
rescoring improves early enrichment over raw Vina — is a **real trend but
statistically underpowered** at n=350, and the report's "+39% / +50%" phrasing
overstates a difference whose confidence interval includes zero. Two new
findings below actually *strengthen* the case on mechanism while tempering it on
significance.

## 1. Statistical power (the main issue)

Paired stratified bootstrap (B=2000, resampling ligands within class) on the
pooled out-of-fold predictions:

| metric | PLIF (95% CI) | raw Vina (95% CI) | Δ (P−V), 95% CI | P(Δ>0) |
|---|---|---|---|---|
| ROC-AUC | 0.583 (0.483–0.678) | 0.635 (0.545–0.720) | −0.054 (−0.175, 0.066) | 0.18 |
| BEDROC(α=20) | 0.468 (0.295–0.623) | 0.337 (0.175–0.502) | +0.128 (−0.107, 0.351) | 0.87 |
| EF₁% | 5.25 (1.75–7.0) | 5.25 (0.0–7.0) | +1.04 (−3.5, 5.25) | 0.54 |
| EF₅% | 3.50 (1.94–5.44) | 2.33 (0.78–4.28) | +1.22 (−1.17, 3.5) | 0.78 |

- **No metric difference reaches 95% significance.** The early-enrichment
  advantage is favored in 87% (BEDROC) and 78% (EF₅%) of bootstrap replicates —
  suggestive, not conclusive. Raw Vina's AUC edge is likewise not significant
  (Δ CI includes 0).
- In raw counts the EF₅% gap is **9 vs 6 actives** in the top 18 compounds, and
  EF₁% is a literal **3-vs-3 tie** in the top 4. These are small integers; the
  percentage framing ("+50%") magnifies a 3-molecule difference.
- **PLIF global discrimination is barely above chance:** AUC 0.583 with lower CI
  0.483 — the interval includes 0.5. The rescorer is an *early-enrichment*
  instrument, not a global classifier, which is consistent with the design goal
  but should be stated plainly.

**Recommendation:** report these CIs in the results table and soften the headline
to "a consistent but not yet significant early-enrichment trend; larger active
sets are needed to confirm." Add bootstrap CIs to `benchmark_results.csv`.

## 2. Decoy/size-bias probe (new — and favorable)

A standard failure mode on DUD-E is a model that learns molecular size rather
than specific contacts.

- DUD-E property matching held: median heavy atoms **28 (actives) vs 27
  (decoys)**, MW **399 vs 391** — no gross size imbalance to exploit.
- **The PLIF rescore is size-decorrelated:** Spearman(rescore, heavy-atoms) =
  **−0.07**. It is not ranking by size.
- **Raw Vina is size-biased:** Spearman(Vina, heavy-atoms) = **+0.51**. This is a
  genuine, citable weakness of the affinity baseline and a real argument for
  interaction-based rescoring that the report does not currently make — worth
  adding.

## 3. Mechanistic interpretability (largely holds, one caveat)

Per-bit active/decoy frequency for the top features:

- **LEU83:HBAcceptor (hinge)** — active 0.16 vs decoy 0.05, **3.2× enriched**.
  The canonical CDK2 hinge H-bond is genuinely discriminative, not a label
  artifact. Strong result.
- **ILE10:Hydrophobic (P-loop)** — 0.38 vs 0.16, **2.4×**. Real.
- **PHE80:Hydrophobic (gatekeeper)** — 0.50 vs 0.38, **only 1.3×**. High gain
  importance but weak class separation; the model likely over-weights it (it
  fires in half of *decoys* too). Flag this — importance ≠ discriminative power.
- 4 of the top 5 bits are hydrophobic (low directional specificity). Only the
  hinge bit is a directional interaction. The mechanistic story rests mainly on
  that one bit plus P-loop packing.

## 4. Other limitations for the record

- Single target (CDK2), single top pose per ligand — no pose-sensitivity or
  cross-target generalization assessment.
- DUD-E's known analog/decoy artifacts are only partially removed by a 2D
  scaffold split; a property-matched external decoy set (e.g. DEKOIS/DrugBank
  actives) would be a stronger test.
- 49 bits × 350 ligands with LightGBM risks mild overfitting even under
  scaffold-CV; the L1-logistic comparator's numbers should be shown alongside
  LightGBM to demonstrate the effect is model-agnostic.

## Bottom line

The pipeline is correct, leakage-controlled, reproducible, and the metrics are
validated. The interpretability check passes on the hinge and P-loop. The
early-enrichment improvement is **directionally real and mechanistically
plausible but not statistically significant at this sample size** — and the
strongest untold part of the story is that the rescorer is *size-decorrelated*
whereas raw Vina is not. Accept the engineering; revise the claims to match the
confidence intervals.
