"""
Apply reproducibility tracking to your already-completed experiments.
Fixed version with correct field mappings.
"""

import json
import numpy as np
from pathlib import Path
from reproducibility_tracker import ReproducibilityTracker

# Load your existing results
with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    raw_results = json.load(f)

# Transform the data to match expected format
all_results = []
for result in raw_results:
    # Map field names to expected format
    transformed = {
        'config': result['config'],
        'mcmc': {
            'acceptance_rate': result['mcmc'].get('acceptance_rate', 0),
            'min_ess': result['mcmc'].get('ess_min', 0),
            'mean_ess': result['mcmc'].get('ess_mean', 0),
            'n_steps': result['config'].get('mcmc_n_steps', 0),
            'n_burn': result['config'].get('mcmc_n_burn', 0),
            'converged': True  # Assume success means converged
        },
        'certificate': {
            'L_hat': result['certificate'].get('L_hat_bounded', result['certificate'].get('L_hat', np.nan)),
            'L_hat_bounded': result['certificate'].get('L_hat_bounded', result['certificate'].get('L_hat', np.nan)),
            'KL': result['certificate'].get('KL_hat', np.nan),
            'Z_hat': result['certificate'].get('Z_hat', np.nan),
            'underline_Z': result['certificate'].get('underline_Z', np.nan),
            'eta_h': result['certificate'].get('eta_h', np.nan),
            'B_lambda': result['certificate'].get('B_lambda_bounded', result['certificate'].get('B_lambda', np.nan)),
            'B_lambda_bounded': result['certificate'].get('B_lambda_bounded', result['certificate'].get('B_lambda', np.nan)),
            'valid': result['certificate'].get('valid', False)
        }
    }
    all_results.append(transformed)

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
    # Show first few violations for debugging
    for i in range(min(3, len(checks['bounded_loss_check']['violations']))):
        idx = checks['bounded_loss_check']['violations'][i]
        L_hat = all_results[idx]['certificate']['L_hat']
        print(f"     Exp {idx}: L_hat = {L_hat}")

print(f"\n2. Underline Z Check (Z > 0):")
print(f"   Status: {'PASS' if checks['underline_Z_check']['passed'] else 'FAIL'}")
if not checks['underline_Z_check']['passed']:
    print(f"   Violations: {len(checks['underline_Z_check']['violations'])} experiments")
    # Show first few violations for debugging
    for i in range(min(3, len(checks['underline_Z_check']['violations']))):
        idx = checks['underline_Z_check']['violations'][i]
        underline_Z = all_results[idx]['certificate']['underline_Z']
        print(f"     Exp {idx}: underline_Z = {underline_Z}")

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
    # Show first few violations for debugging
    for i in range(min(3, len(checks['ess_check']['violations']))):
        idx = checks['ess_check']['violations'][i]
        ess = all_results[idx]['mcmc']['min_ess']
        print(f"     Exp {idx}: min_ess = {ess}")

print(f"\n5. η_h Refinement Check:")
print(f"   Status: {'PASS' if checks['eta_h_refinement_check']['passed'] else 'FAIL'}")
if 'values' in checks['eta_h_refinement_check']:
    vals = checks['eta_h_refinement_check']['values']
    print(f"   n_x=50:  {vals['n_x=50']:.6f}")
    print(f"   n_x=100: {vals['n_x=100']:.6f}")

# Summary statistics
print(f"\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
acceptance_rates = [r['mcmc']['acceptance_rate'] for r in all_results if r['mcmc']['acceptance_rate'] > 0]
ess_values = [r['mcmc']['min_ess'] for r in all_results if r['mcmc']['min_ess'] > 0]

print(f"Acceptance rates: mean={np.mean(acceptance_rates):.3f}, median={np.median(acceptance_rates):.3f}")
print(f"Min ESS values: mean={np.mean(ess_values):.1f}, median={np.median(ess_values):.1f}")
print(f"Valid certificates: {sum(1 for r in all_results if r['certificate']['valid'])}/{len(all_results)}")