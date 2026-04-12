# JARVIS-50 QE Verification Checklist

This checklist applies the repo-wide protocol in
`docs/validation/DFT_verification_protocol.md` to the current
`reports/dft_qe_jarvis_50_mix` campaign.

## Workflow declaration

Current baseline from `settings.json`:

- `calculation = relax`
- `kspacing = 0.25`
- `ecutwfc = 100 Ry`
- `ecutrho = 1080 Ry`
- `degauss = 0.02 Ry`
- pseudopotential map: `pseudos.json`

## Campaign status

- [x] Candidate manifest committed in `manifest.csv`
- [x] QE settings committed in `settings.json`
- [x] Runner scripts committed for local and Slurm execution
- [ ] Per-candidate `pw.out` reviewed for SCF and ionic convergence
- [ ] Final max-force summary captured for every completed relaxation
- [ ] Cutoff convergence reruns completed on top 5 to 10 candidates
- [ ] K-point convergence reruns completed on top 5 to 10 candidates
- [ ] Competing-phase hull or compatible reference workflow documented
- [ ] Spin/U sensitivity completed for open-shell transition-metal oxides
- [ ] Final evidence tier assigned per candidate
- [ ] Phonons completed for finalists promoted beyond `DFT-verified`

## Claim guardrails

Until the unchecked items above are complete:

- candidates remain `ML-screened` or `QE-relaxed`
- the campaign may be described as `provisional evidence`
- the campaign must **not** be described as `DFT-verified`

## Immediate next actions

1. Parse all completed `pw.out` files and record convergence status.
2. Rerun the top 5 to 10 candidates with tighter cutoffs and denser k-point spacing.
3. Build or document the competing-phase reference hull with the same workflow conventions.
4. Run phonons only on the finalists that survive the hull and sensitivity screens.
