Core suggestion: build a “CrystalFold” stack (structure encoder + calibrated decision layer)

Your goal isn’t “best MAE.” It’s high-confidence top-k selection with reliable risk control under strong target skew (most near 0) and chemically-aware splits.

1) Input strategy (what users will actually have)

You need to support two entry modes:

A) Composition-only mode (fast triage, no CIF yet)

Input: formula + oxidation state guess (optional).

Output: rough stability prior + “need structure” flag.

Model: composition transformer / ElemNet-style MLP with learned element embeddings.

B) Structure mode (CIF → atomic graph; final screening)

Input: crystal structure (CIF), periodic.

Output: calibrated stability probability + expected Ehull + uncertainty interval.

Why: in industry, many candidates start as composition ideas before a finalized structure.

2) Model architecture: AlphaFold-inspired, but crystals

AlphaFold’s winning pattern is: (i) rich pair representation + (ii) attention + (iii) recycling + (iv) geometry-aware updates. Translate that to periodic crystals:

Recommended “best ROI” structure encoder

A periodic graph transformer with local geometry + “pair” features

Node tokens: atoms (element embedding + oxidation hints + site features).

Edge/pair tokens: distance RBF + (optional) angular features + periodic image vector.

Blocks:

Local message passing (efficient) + global attention pooling (for interpretability).

Recycling: run the same block 2–3 times, feeding back updated embeddings (cheap way to mimic AlphaFold refinement without huge models).

If you can afford more sophistication (still CPU-safe with careful sizing):

Add angle/line-graph features (ALIGNN-lite idea): big boost for oxides.

Add Voronoi coordination features (cheap preprocessing) to reduce catastrophic outliers.

Avoid full-blown SE(3) equivariant monsters unless you’ve benchmarked CPU latency; they can overshoot your 100ms constraint.

3) Targets and losses: stop treating Ehull as a symmetric regression

Your data is heavily skewed toward 0 and your product decision is threshold-based (0.05 / 0.10 eV). So change the training objective:

Multi-head outputs (single forward pass)

μ = predicted Ehull

σ = uncertainty scale

p_stable = P(Ehull < 0.05) (explicit classification head)

Loss (high ROI)

Regression: robust (Huber / quantile) or heavy-tailed likelihood (Student-t) instead of pure Gaussian NLL.

Classification: weighted BCE for stability threshold.

Optional: add an auxiliary target if available (formation energy), but don’t let it dominate.

This directly optimizes what the business cares about: top-k stable recall and false-positive control.

4) Uncertainty you can trust (calibration matters more than fancy UQ)

Your model already predicts σ, but you need it to be decision-grade.

Practical uncertainty stack (CPU-friendly)

Train single model with heteroscedastic head (σ).

Add post-hoc calibration:

temperature scaling for σ (global)

conformal prediction for interval coverage (cheap, strong guarantees)

Risk score (what customers want)

Return:

P(stable < 0.05)

P(metastable < 0.10)

Lower confidence bound: LCB = μ + kσ (k chosen to control false positives)

Then define:

KEEP if P(stable) ≥ 0.9 and LCB < 0.05

MAYBE if probabilities are mixed / uncertainty high

KILL if P(metastable) ≤ 0.1 or LCB > 0.10

This is more robust than ranking by μ alone.

5) Data strategy: your biggest gains will come from fixing tail behavior

You have heavy-tailed catastrophic errors. Those are often:

OOD chemistries vs train families

weird structures / bad relaxation artifacts

rare coordinations underrepresented in train

Do this:

Hard-example mining: upweight top 1–5% highest-error samples each epoch (or focal-style weighting).

Balanced sampling by Ehull bins: don’t let the near-0 mass dominate.

OOD detection: flag when composition family is novel vs train split (cheap heuristic + learned embedding distance).

Next-level (massive ROI): active learning loop

Each week: model picks “high uncertainty + high upside” candidates → run DFT on small batch → retrain.
This is the closest analog to AlphaFold’s “data flywheel.”

6) Evaluation that matches the product

Stop judging by MAE/RMSE alone.

Track:

Recall@K for stable (Ehull < 0.05)

Precision@K

False-stable rate (pred stable but true > 0.10)

Calibration: coverage of predicted intervals, ECE-style for P(stable)

Your model is “better” if it saves 30% of DFT calls at the same discovery rate.

7) Deployment design (B2B-ready)

Precompute graph features (neighbors, Voronoi stats) once per CIF.

Export to ONNX / TorchScript; keep model small (<50MB).

Batch inference for 100 structures (CPU vectorization).

Add explanation:

attention pooling weights + simple per-element contribution summaries.

Recommended build plan (prioritized)

Add multi-head (μ, σ, p_stable) + switch loss to robust/heavy-tailed + balanced sampling.

Implement calibration (temperature for σ + conformal intervals) + decision policy (KEEP/MAYBE/KILL).

Upgrade encoder to CrystalFold-lite: recycling + angle/Voronoi features (choose one first).

Add hard-example mining + OOD flags.

Set up active learning with small DFT batches.

If you want, I can propose an exact architecture spec (layer counts, dims, pooling, losses) that stays under 50MB and should fit <100ms CPU, using your current CGCNN as the baseline.