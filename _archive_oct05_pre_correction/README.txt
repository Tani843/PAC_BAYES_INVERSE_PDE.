Pre-correction archive (Oct 5, 2025)
========================================
This folder contains ALL results computed before correcting the Gibbs posterior scaling formula.

ISSUE: All results used incorrect scaling exp(-λ L) instead of exp(-λn L)
IMPACT: Missing factor of n made PAC-Bayes certificates meaningless (bounds too large)

Contents:
- results_full_section_a: Nearly complete Section A run (1746/1728 experiments)
- results_main_focused_72: Focused main results (73 experiments)
- results_baseline: Classical baseline results
- results_extended: Extended experiment results  
- results_seed202: Seed 202 specific results
- results_seed303: Seed 303 specific results

ALL archived for reference but scientifically invalid due to incorrect mathematical formulation.
Workspace is now completely clean for corrected Phase 2 execution.

