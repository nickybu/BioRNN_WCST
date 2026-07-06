# Critical re-analysis of Liu & Wang (2024) — BioRNN WCST

This branch adds an independent analysis + retraining toolkit and a manuscript
critically re-examining two central claims of Liu & Wang (2024, Nat Commun 15:6497,
doi:10.1038/s41467-024-50501-y): that the task rule is maintained by PFC attractors,
and that SST interneurons uniquely gate the angle between rule subspaces.

## Contents
- `analysis/analysis.py` — analysis pipeline (loading, simulation, rule decoding,
  subspace principal angle, optogenetic-style silencing, inter-modular lesion,
  autonomous-dynamics attractor test, recurrent-vs-feedforward current decomposition).
- `analysis/train_local.py` — self-contained, resumable curriculum training harness
  (cluster paths removed; stable running-average curriculum criterion; weight+optimizer
  +stage checkpointing).
- `analysis/ablations.py` — ablation configs (drop-PV, drop-dendrites, shrink control).
- `analysis/run_ablations.py` — multi-seed ablation launcher with per-network checkpointing.
- `manuscript/manuscript.md`, `manuscript/manuscript.pdf` — the write-up.

## Findings (analysis on the 151 released pretrained networks)
1. **Rule locus** — maintained in the PFC↔SR *loop*, not either module alone
   (autonomous attractor collapses in both modules after inter-modular lesion;
   cross-module silencing is symmetric).
2. **Dynamics vs connectivity** — the rule is held by *recurrent dynamics*: decodable
   at ~0.91 during the pre-card inter-trial interval in 36/36 networks, when no stimulus
   is present.
3. **SST specificity** — SST is *not* the unique controller of the SR rule-subspace angle.
   PV silencing collapses it more; across manipulations the angle collapse is largely a
   generic readout of task performance (Pearson r≈0.86, R²≈0.74).

## Status
Ablation retraining (drop-PV, drop-dendrites, emergence-during-learning, transfer capstone)
is in progress; harness and configs are included and tested.
