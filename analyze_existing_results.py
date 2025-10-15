"""
Apply reproducibility tracking to your already-completed experiments.
"""

import json
from pathlib import Path
from reproducibility_tracker import ReproducibilityTracker

# Load your existing results
with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    all_results = json.load(f)

# Initialize tracker
tracker = ReproducibilityTracker(Path('reproducibility_logs'))

# Save environment info
tracker.save_environment_info()

# Run pre-publication checks on your existing data
checks = tracker.run_prepublish_checks(all_results)

# Print detailed check results
print("\n" + "="*60)
print("DETAILED PRE-PUBLICATION CHECK RESULTS")
print("="*60)

print(f"\n1. Bounded Loss Check (ℓ ∈ (0,1)):")
print(f"   Status: {'PASS' if checks['bounded_loss_check']['passed'] else 'FAIL'}")
if not checks['bounded_loss_check']['passed']:
    print(f"   Violations: {len(checks['bounded_loss_check']['violations'])} experiments")

print(f"\n2. Underline Z Check (Z > 0):")
print(f"   Status: {'PASS' if checks['underline_Z_check']['passed'] else 'FAIL'}")
if not checks['underline_Z_check']['passed']:
    print(f"   Violations: {len(checks['underline_Z_check']['violations'])} experiments")

print(f"\n3. Acceptance Rate Check ([0.2, 0.5]):")
print(f"   Status: {'PASS' if checks['acceptance_check']['passed'] else 'FAIL'}")
if not checks['acceptance_check']['passed']:
    print(f"   Violations: {len(checks['acceptance_check']['violations'])} experiments")
    # Show some examples
    violations = checks['acceptance_check']['violations'][:5]
    for v in violations:
        acc = all_results[v]['mcmc'].get('acceptance_rate', 0)
        print(f"     Exp {v}: acceptance = {acc:.3f}")

print(f"\n4. ESS Check (≥ 50):")
print(f"   Status: {'PASS' if checks['ess_check']['passed'] else 'FAIL'}")
if not checks['ess_check']['passed']:
    print(f"   Violations: {len(checks['ess_check']['violations'])} experiments")

print(f"\n5. η_h Refinement Check:")
print(f"   Status: {'PASS' if checks['eta_h_refinement_check']['passed'] else 'FAIL'}")
if 'values' in checks['eta_h_refinement_check']:
    vals = checks['eta_h_refinement_check']['values']
    print(f"   n_x=50:  {vals['n_x=50']:.6f}")
    print(f"   n_x=100: {vals['n_x=100']:.6f}")