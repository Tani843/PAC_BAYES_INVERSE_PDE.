"""
Apply reproducibility tracking to your already-completed experiments.
Robust version that handles failed experiments.
"""

import json
import numpy as np
from pathlib import Path
from reproducibility_tracker import ReproducibilityTracker

# Load your existing results
with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    raw_results = json.load(f)

# Filter successful experiments and transform the data
all_results = []
failed_experiments = []

for i, result in enumerate(raw_results):
    if result.get('status') == 'success' and 'mcmc' in result and 'certificate' in result:
        # Map field names to expected format
        transformed = {
            'config': result['config'],
            'mcmc': {
                'acceptance_rate': result['mcmc'].get('acceptance_rate', 0),
                'min_ess': result['mcmc'].get('ess_min', 0),
                'mean_ess': result['mcmc'].get('ess_mean', 0),
                'n_steps': result['mcmc'].get('total_steps', result['config'].get('mcmc_n_steps', 0)),
                'n_burn': result['config'].get('mcmc_n_burn', 0),
                'converged': result['mcmc'].get('converged', True)
            },
            'certificate': {
                'L_hat': result['certificate'].get('L_hat', np.nan),
                'L_hat_bounded': result['certificate'].get('L_hat_bounded', np.nan),
                'KL': result['certificate'].get('KL', np.nan),
                'Z_hat': result['certificate'].get('Z_hat', np.nan),
                'underline_Z': result['certificate'].get('underline_Z', np.nan),
                'eta_h': result['certificate'].get('eta_h', np.nan),
                'B_lambda': result['certificate'].get('B_lambda', np.nan),
                'B_lambda_bounded': result['certificate'].get('B_lambda_bounded', np.nan),
                'valid': result['certificate'].get('valid', False)
            }
        }
        all_results.append(transformed)
    else:
        failed_experiments.append({
            'index': i,
            'experiment_id': result.get('experiment_id', 'unknown'),
            'status': result.get('status', 'unknown'),
            'error': result.get('error', 'No error message')
        })

print(f"="*60)
print(f"EXPERIMENT SUCCESS/FAILURE SUMMARY")
print(f"="*60)
print(f"Total experiments: {len(raw_results)}")
print(f"Successful experiments: {len(all_results)}")
print(f"Failed experiments: {len(failed_experiments)}")
print(f"Success rate: {len(all_results)/len(raw_results)*100:.1f}%")

if failed_experiments:
    print(f"\nFirst few failed experiments:")
    for exp in failed_experiments[:5]:
        print(f"  {exp['index']}: {exp['experiment_id']} - {exp['error'][:100]}...")

# Initialize tracker
tracker = ReproducibilityTracker(Path('reproducibility_logs'))

# Save environment info
tracker.save_environment_info()

# Modified checks for the actual data structure
def run_modified_checks(all_results):
    """Run checks appropriate for the actual data structure."""
    checks = {
        'total_experiments': len(all_results),
        'acceptance_check': {'passed': True, 'violations': []},
        'ess_check': {'passed': True, 'violations': []},
        'eta_h_refinement_check': {'passed': True},
        'certificate_validity_check': {'passed': True, 'valid_count': 0}
    }
    
    acc_lo, acc_hi = 0.2, 0.5
    ess_threshold = 50
    
    # Check each experiment
    for i, result in enumerate(all_results):
        cert = result.get('certificate', {})
        mcmc = result.get('mcmc', {})
        config = result.get('config', {})
        
        # 1. Check acceptance ∈ [acc_lo, acc_hi]
        try:
            acc = float(mcmc.get('acceptance_rate', 0))
        except Exception:
            acc = 0.0
        if not (acc_lo <= acc <= acc_hi):
            checks['acceptance_check']['passed'] = False
            checks['acceptance_check']['violations'].append(i)
        
        # 2. Check ESS ≥ threshold
        try:
            ess = float(mcmc.get('min_ess', 0))
        except Exception:
            ess = 0.0
        if ess < ess_threshold:
            checks['ess_check']['passed'] = False
            checks['ess_check']['violations'].append(i)
        
        # 3. Count valid certificates
        if cert.get('valid', False):
            checks['certificate_validity_check']['valid_count'] += 1
    
    # 4. Check η_h decreases with refinement
    eta_by_nx = {}
    for result in all_results:
        cfg = result.get('config', {})
        cert = result.get('certificate', {})
        nx = cfg.get('n_x')
        eta = cert.get('eta_h', np.nan)
        try:
            nx_int = int(nx)
            eta_float = float(eta)
        except Exception:
            continue
        if not np.isnan(eta_float):
            eta_by_nx.setdefault(nx_int, []).append(eta_float)
    
    if 50 in eta_by_nx and 100 in eta_by_nx and len(eta_by_nx[50]) and len(eta_by_nx[100]):
        mean_50 = float(np.mean(eta_by_nx[50]))
        mean_100 = float(np.mean(eta_by_nx[100]))
        checks['eta_h_refinement_check']['passed'] = (mean_100 <= mean_50)
        checks['eta_h_refinement_check']['values'] = {
            'n_x=50': mean_50,
            'n_x=100': mean_100
        }
    
    return checks

# Run modified checks
checks = run_modified_checks(all_results)

# Print detailed check results
print(f"\n" + "="*60)
print("PRE-PUBLICATION CHECK RESULTS")
print("="*60)

print(f"\n1. Acceptance Rate Check ([0.2, 0.5]):")
print(f"   Status: {'PASS' if checks['acceptance_check']['passed'] else 'FAIL'}")
if not checks['acceptance_check']['passed']:
    print(f"   Violations: {len(checks['acceptance_check']['violations'])} experiments")
    violations = checks['acceptance_check']['violations'][:5]
    for v in violations:
        acc = all_results[v]['mcmc'].get('acceptance_rate', 0)
        print(f"     Exp {v}: acceptance = {acc:.3f}")
else:
    print(f"   ✓ All {len(all_results)} successful experiments have acceptance rates in [0.2, 0.5]")

print(f"\n2. ESS Check (≥ 50):")
print(f"   Status: {'PASS' if checks['ess_check']['passed'] else 'FAIL'}")
if not checks['ess_check']['passed']:
    print(f"   Violations: {len(checks['ess_check']['violations'])} experiments")
    for i in range(min(5, len(checks['ess_check']['violations']))):
        idx = checks['ess_check']['violations'][i]
        ess = all_results[idx]['mcmc']['min_ess']
        print(f"     Exp {idx}: min_ess = {ess:.1f}")
else:
    print(f"   ✓ All {len(all_results)} successful experiments have ESS ≥ 50")

print(f"\n3. η_h Refinement Check:")
print(f"   Status: {'PASS' if checks['eta_h_refinement_check']['passed'] else 'FAIL'}")
if 'values' in checks['eta_h_refinement_check']:
    vals = checks['eta_h_refinement_check']['values']
    print(f"   n_x=50:  {vals['n_x=50']:.6f}")
    print(f"   n_x=100: {vals['n_x=100']:.6f}")
    if checks['eta_h_refinement_check']['passed']:
        print(f"   ✓ η_h decreases with mesh refinement as expected")

print(f"\n4. Certificate Validity:")
valid_count = checks['certificate_validity_check']['valid_count']
total_count = checks['total_experiments']
print(f"   Valid certificates: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
if valid_count == total_count:
    print(f"   ✓ All successful experiments produced valid certificates")

# Summary statistics
print(f"\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

acceptance_rates = [r['mcmc']['acceptance_rate'] for r in all_results if r['mcmc']['acceptance_rate'] > 0]
ess_values = [r['mcmc']['min_ess'] for r in all_results if r['mcmc']['min_ess'] > 0]
l_hat_values = [r['certificate']['L_hat'] for r in all_results if not np.isnan(r['certificate']['L_hat'])]

print(f"Successful experiments: {len(all_results)}")
print(f"Acceptance rates: mean={np.mean(acceptance_rates):.3f}, median={np.median(acceptance_rates):.3f}")
print(f"  Range: [{np.min(acceptance_rates):.3f}, {np.max(acceptance_rates):.3f}]")
print(f"Min ESS values: mean={np.mean(ess_values):.1f}, median={np.median(ess_values):.1f}")
print(f"  Range: [{np.min(ess_values):.1f}, {np.max(ess_values):.1f}]")
print(f"L_hat values: mean={np.mean(l_hat_values):.1f}, median={np.median(l_hat_values):.1f}")

# Parameter coverage
s_values = sorted(set(r['config']['s'] for r in all_results))
sigma_values = sorted(set(r['config']['sigma'] for r in all_results))
nx_values = sorted(set(r['config']['n_x'] for r in all_results))
lambda_values = sorted(set(r['config']['lambda'] for r in all_results))

print(f"\nParameter coverage (successful experiments):")
print(f"  s: {s_values}")
print(f"  σ: {sigma_values}")
print(f"  n_x: {nx_values}")
print(f"  λ: {len(lambda_values)} values from {min(lambda_values):.3f} to {max(lambda_values):.3f}")

# Final recommendation
print(f"\n" + "="*60)
print("REPRODUCIBILITY ASSESSMENT")
print("="*60)

all_checks_pass = (
    checks['acceptance_check']['passed'] and
    checks['ess_check']['passed'] and
    checks['eta_h_refinement_check']['passed'] and
    valid_count == total_count
)

if all_checks_pass:
    print("✓ PUBLICATION READY")
    print("  All quality checks pass for successful experiments.")
    print("  The results demonstrate robust MCMC sampling and valid certificates.")
else:
    print("⚠ REVIEW REQUIRED")
    print("  Some quality checks failed. Review violations before publication.")

print(f"\nSuccess rate: {len(all_results)}/{len(raw_results)} = {len(all_results)/len(raw_results)*100:.1f}%")
if len(failed_experiments) > 0:
    print(f"Consider investigating the {len(failed_experiments)} failed experiments.")