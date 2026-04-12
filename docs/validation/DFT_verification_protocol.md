# DFT Verification Protocol

**Document ID:** VV-DFT-001  
**Version:** 1.0  
**Effective Date:** 2026-04-13  
**System:** CathodeScreen candidate verification  
**Classification:** Scientific Verification

---

## 1. Purpose

This protocol defines what CathodeScreen is allowed to claim after Quantum ESPRESSO (QE)
follow-up. The goal is to separate:

- `ML-screened` candidates, which are ranked by the ensemble and uncertainty policy
- `DFT-checked` candidates, which were relaxed under one declared QE workflow
- `DFT-verified` candidates, which passed numerical convergence, reference-hull, and
  robustness checks

This is a project acceptance protocol, not a universal law of DFT. The numerical thresholds
below are CathodeScreen policy thresholds chosen to support credible shortlist claims.

## 2. External Basis

This protocol is grounded in the following primary or official sources:

- Quantum ESPRESSO `pw.x` input documentation, which defines the core numerical controls
  used here: `ecutwfc`, `ecutrho`, `K_POINTS`, `conv_thr`, `etot_conv_thr`,
  and `forc_conv_thr`.
- Quantum ESPRESSO PHonon user guide, which states that phonon calculations are performed
  after a converged ground-state calculation from `pw.x`.
- Materials Project methodology documentation, which describes the consistent use of
  GGA/GGA+U settings and the GGA/GGA+U/r2SCAN mixing scheme for thermodynamics.
- Phonopy documentation and standard phonon conventions, where imaginary modes are reported
  as negative frequencies.

## 3. Evidence Tiers

| Tier | Label | Minimum evidence | Allowed claim |
|---|---|---|---|
| T0 | `ML-screened` | Ensemble prediction, uncertainty, and action policy only | Candidate is worth DFT follow-up |
| T1 | `QE-relaxed` | Candidate relaxed with one declared QE workflow and stored outputs | Candidate was relaxed under the stated DFT setup |
| T2 | `DFT-hull-checked` | T1 plus consistent competing-phase or compatible reference-hull evaluation | Candidate has provisional first-principles support under one thermodynamic workflow |
| T3 | `DFT-verified` | T2 plus numerical convergence and spin/U sensitivity checks | Candidate remains favorable within the declared DFT workflow |
| T4 | `Phonon-screened` | T3 plus dynamic-stability screening with phonons | Candidate shows no material dynamic instability within the stated phonon check |
| T5 | `Experimentally supported` | T4 plus external or in-house experimental support | Candidate has support beyond computation |

Public-facing pages should avoid the word `verified` unless a candidate has reached at least T3.

## 4. Required Artifacts Per Campaign

Every QE campaign must store:

- workflow declaration: functional, `+U` policy, smearing, pseudopotential family,
  `ecutwfc`, `ecutrho`, `kspacing` or explicit k-mesh, relaxation mode
- pseudopotential manifest and exact filenames
- candidate manifest and immutable ranking snapshot used to launch the campaign
- full QE inputs and outputs for every candidate and every reference phase
- convergence summary for cutoffs and k-point density
- spin/U sensitivity summary for open-shell transition-metal systems
- hull reconstruction inputs and outputs, or an explicit statement that a Materials Project
  mixing scheme was used
- phonon inputs and outputs for any candidate promoted to T4
- provenance: commit SHA, script versions, timestamps, and operator

## 5. Minimum Acceptance Gates

### 5.1 Relaxation completion

For a candidate to reach T1:

- QE must finish without an SCF or ionic-convergence failure.
- Final ionic step must satisfy the declared relaxation thresholds.
- Final maximum force must be recorded and should be `<= 0.02 eV/Angstrom` as CathodeScreen
  policy.

### 5.2 Numerical convergence

For a candidate to reach T3:

- Tighten the plane-wave basis once relative to the baseline workflow.
- Tighten the k-point density once relative to the baseline workflow.
- Accept only if both checks remain numerically stable by CathodeScreen policy:
  - `|Delta E_total| <= 0.002 eV/atom`
  - `|Delta E_hull| <= 0.005 eV/atom`
  - final stable/unstable classification unchanged

These are project thresholds used to decide whether a result is robust enough for shortlist
claims.

### 5.3 Reference-hull consistency

For a candidate to reach T2 or higher:

- The candidate must be compared against a chemically relevant competing-phase set.
- Candidate and reference phases must use the same pseudopotential family and the same
  functional/U/spin conventions.
- If external reference energies are mixed in, the mixing rule must be explicit and
  compatible with a published scheme such as the Materials Project GGA/GGA+U/r2SCAN
  thermodynamic framework.

Single-candidate relaxations without a reference hull do not count as `DFT-verified`.

### 5.4 Spin and Hubbard-U sensitivity

For open-shell transition-metal oxides:

- Run at least two distinct initial magnetic configurations.
- Use one declared Hubbard-U table across candidate and reference phases.
- If the material remains promising after the baseline workflow, run one higher-fidelity
  sensitivity check, preferably an MP-compatible `r2SCAN` single-point or follow-up relax on
  the final shortlist.
- Accept only if the candidate remains within the same practical decision bucket after these
  checks.

### 5.5 Phonon screening

For T4:

- Start from a converged relaxed structure.
- Run phonons only after the ground-state calculation is numerically stable.
- For shortlist-level credibility, Gamma-point phonons are the minimum screen.
- For publication-grade claims, prefer a fuller q-grid or finite-displacement dispersion.
- CathodeScreen policy treats frequencies below `-0.5 THz` after standard cleanup as a
  material imaginary-mode failure that blocks T4.

### 5.6 Reproducibility

Every promoted candidate must be reproducible from committed metadata:

- exact input structure
- exact QE inputs
- exact pseudopotential set
- exact workflow settings
- final relaxed structure
- parsed summary of energies and status

## 6. Current Repository Baseline

The repository already contains a prepared QE audit bundle in
`reports/dft_qe_jarvis_50_mix` with:

- `settings.json`
- `manifest.csv`
- `check_pseudos.py`
- local and Slurm runner scripts

Current recorded baseline settings are:

- `calculation = relax`
- `kspacing = 0.25`
- `ecutwfc = 100 Ry`
- `ecutrho = 1080 Ry`
- `degauss = 0.02 Ry`
- pseudopotential map from `pseudos.json`

This is enough to claim that a QE audit workflow is configured. It is **not** enough by
itself to claim `DFT-verified` candidates. Missing evidence still includes:

- per-candidate convergence summaries
- explicit competing-phase hull reconstruction or declared mixed-reference workflow
- spin/U sensitivity records for open-shell systems
- phonon results for promoted finalists

## 7. Immediate Execution Plan

The first credibility milestone for this project is:

1. keep the current ML shortlist labeled as `ML-screened`
2. promote only completed QE relaxations to `QE-relaxed`
3. run convergence checks on the top 5 to 10 candidates
4. rebuild or document the reference hull for the same chemistry
5. run phonons only on the surviving finalists

Until steps 3 and 4 are complete, the screening pack should be described as
`provisional evidence`, not `proof`.

## 8. References

- Quantum ESPRESSO `pw.x` input description: https://www.quantum-espresso.org/Doc/INPUT_PW.html
- Quantum ESPRESSO PHonon user guide: https://www.quantum-espresso.org/Doc/user_guide_PDF/ph_user_guide.pdf
- Materials Project documentation: https://docs.materialsproject.org/
- Phonopy documentation: https://phonopy.github.io/phonopy/
