#!/usr/bin/env python3
"""
Verify Phase 2 complete results against specifications
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def verify_phase2_results(results_file):
    """Comprehensive verification of Phase 2 experimental results."""
    
    print("="*80)
    print("PHASE 2 RESULTS VERIFICATION")
    print("="*80)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n✓ Loaded {len(results)} experiments from {results_file}")
    
    # Initialize counters
    stats = {
        'total': len(results),
        'errors': 0,
        'valid_certificates': 0,
        'invalid_certificates': 0,
        'converged_mcmc': 0,
        'ess_above_200': 0,
        'ess_above_100': 0,
        'ess_above_50': 0,
        'ess_below_50': 0,
        'kl_zero': 0,
        'kl_nonzero': 0,
        'missing_fields': []
    }
    
    # Expected experimental grid
    expected_configs = {
        's': [3, 5],
        'placement_type': ['fixed', 'shifted'],
        'sigma': [0.05, 0.10, 0.20],
        'n_x': [50, 100],
        'T': [0.3, 0.5],
        'lambda': [0.5, 1.0, 2.0],
        'm': [3, 5],
        'n_t': [50, 100],
        'seed': [101, 202, 303]
    }
    
    # Track what we actually have
    actual_configs = defaultdict(set)
    
    # Detailed diagnostics
    issues = []
    ess_values = []
    kl_values = []
    certificate_margins = []
    runtimes = []
    
    print("\n" + "="*80)
    print("VALIDATING EACH EXPERIMENT")
    print("="*80)
    
    for i, result in enumerate(results):
        exp_id = result.get('experiment_id', f'experiment_{i}')
        
        # Check for errors
        if 'error' in result:
            stats['errors'] += 1
            issues.append(f"❌ {exp_id}: ERROR - {result['error']}")
            continue
        
        # Verify required fields
        required_fields = ['config', 'dataset', 'mcmc', 'certificate', 
                          'posterior_summary', 'performance']
        missing = [f for f in required_fields if f not in result]
        if missing:
            stats['missing_fields'].append((exp_id, missing))
            issues.append(f"⚠️  {exp_id}: Missing fields {missing}")
        
        # Extract data
        config = result.get('config', {})
        mcmc = result.get('mcmc', {})
        cert = result.get('certificate', {})
        perf = result.get('performance', {})
        
        # Track actual configurations
        for key in expected_configs:
            if key in config:
                actual_configs[key].add(config[key])
        
        # Certificate validity
        if cert.get('valid', False):
            stats['valid_certificates'] += 1
            # Calculate margin
            B_lambda = cert.get('B_lambda', 0)
            L_hat = cert.get('L_hat', 0)
            margin = B_lambda - L_hat
            certificate_margins.append(margin)
            
            if margin < 0:
                issues.append(f"🚨 {exp_id}: INVALID margin {margin:.6f}")
        else:
            stats['invalid_certificates'] += 1
            issues.append(f"❌ {exp_id}: Invalid certificate")
        
        # MCMC convergence
        if mcmc.get('converged', False):
            stats['converged_mcmc'] += 1
        
        # ESS analysis
        ess_min = mcmc.get('ess_min', 0)
        ess_values.append(ess_min)
        
        if ess_min >= 200:
            stats['ess_above_200'] += 1
        elif ess_min >= 100:
            stats['ess_above_100'] += 1
        elif ess_min >= 50:
            stats['ess_above_50'] += 1
        else:
            stats['ess_below_50'] += 1
        
        # KL divergence
        kl = cert.get('KL', 0)
        kl_values.append(kl)
        
        if kl == 0.0:
            stats['kl_zero'] += 1
        else:
            stats['kl_nonzero'] += 1
        
        # Runtime
        runtime = perf.get('runtime', 0)
        runtimes.append(runtime)
    
    # Print configuration coverage
    print("\n" + "="*80)
    print("CONFIGURATION COVERAGE")
    print("="*80)
    
    coverage_complete = True
    for param, expected_values in expected_configs.items():
        actual_values = sorted(actual_configs[param])
        expected_sorted = sorted(expected_values)
        
        if set(actual_values) == set(expected_sorted):
            print(f"✓ {param:8s}: {actual_values} (complete)")
        else:
            print(f"❌ {param:8s}: Expected {expected_sorted}, got {actual_values}")
            coverage_complete = False
    
    # Expected total experiments
    expected_total = 1
    for values in expected_configs.values():
        expected_total *= len(values)
    
    print(f"\n{'='*80}")
    print(f"Expected total experiments: {expected_total}")
    print(f"Actual experiments: {stats['total']}")
    if stats['total'] == expected_total:
        print("✓ Complete experimental grid!")
    else:
        print(f"❌ Missing {expected_total - stats['total']} experiments")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n📊 Overall Status:")
    print(f"   Total experiments: {stats['total']}")
    print(f"   Errors: {stats['errors']} ({stats['errors']/stats['total']*100:.1f}%)")
    print(f"   Successful: {stats['total'] - stats['errors']} ({(stats['total']-stats['errors'])/stats['total']*100:.1f}%)")
    
    if stats['missing_fields']:
        print(f"\n⚠️  Experiments with missing fields: {len(stats['missing_fields'])}")
    
    print(f"\n🏆 Certificate Validity:")
    print(f"   Valid certificates: {stats['valid_certificates']}/{stats['total']} ({stats['valid_certificates']/stats['total']*100:.1f}%)")
    print(f"   Invalid certificates: {stats['invalid_certificates']}")
    
    if certificate_margins:
        print(f"   Certificate margin (B_λ - L̂):")
        print(f"      Mean: {np.mean(certificate_margins):.4f}")
        print(f"      Min: {np.min(certificate_margins):.4f}")
        print(f"      Max: {np.max(certificate_margins):.4f}")
    
    print(f"\n🔬 MCMC Performance:")
    print(f"   Converged (ESS≥200): {stats['converged_mcmc']}/{stats['total']} ({stats['converged_mcmc']/stats['total']*100:.1f}%)")
    
    print(f"\n📈 ESS Distribution:")
    print(f"   ESS ≥ 200: {stats['ess_above_200']} ({stats['ess_above_200']/stats['total']*100:.1f}%)")
    print(f"   ESS 100-199: {stats['ess_above_100']} ({stats['ess_above_100']/stats['total']*100:.1f}%)")
    print(f"   ESS 50-99: {stats['ess_above_50']} ({stats['ess_above_50']/stats['total']*100:.1f}%)")
    print(f"   ESS < 50: {stats['ess_below_50']} ({stats['ess_below_50']/stats['total']*100:.1f}%)")
    
    if ess_values:
        print(f"\n   ESS Statistics:")
        print(f"      Mean: {np.mean(ess_values):.1f}")
        print(f"      Median: {np.median(ess_values):.1f}")
        print(f"      Min: {np.min(ess_values):.1f}")
        print(f"      Max: {np.max(ess_values):.1f}")
    
    print(f"\n🔧 KL Divergence:")
    print(f"   KL = 0 (Hoeffding failed): {stats['kl_zero']} ({stats['kl_zero']/stats['total']*100:.1f}%)")
    print(f"   KL > 0 (Hoeffding worked): {stats['kl_nonzero']} ({stats['kl_nonzero']/stats['total']*100:.1f}%)")
    
    if kl_values:
        nonzero_kl = [k for k in kl_values if k > 0]
        if nonzero_kl:
            print(f"   Non-zero KL statistics:")
            print(f"      Mean: {np.mean(nonzero_kl):.4f}")
            print(f"      Median: {np.median(nonzero_kl):.4f}")
    
    if runtimes:
        print(f"\n⏱️  Runtime:")
        print(f"   Mean: {np.mean(runtimes):.1f}s ({np.mean(runtimes)/60:.1f} min)")
        print(f"   Median: {np.median(runtimes):.1f}s")
        print(f"   Total: {np.sum(runtimes)/3600:.1f} hours")
    
    # Print issues if any
    if issues:
        print("\n" + "="*80)
        print(f"ISSUES FOUND ({len(issues)})")
        print("="*80)
        for issue in issues[:20]:  # Show first 20
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues)-20} more")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    success = (
        stats['total'] == expected_total and
        stats['errors'] == 0 and
        stats['valid_certificates'] == stats['total'] and
        coverage_complete
    )
    
    if success:
        print("✅ ALL SPECIFICATIONS MET!")
        print("   ✓ Complete experimental grid")
        print("   ✓ No errors")
        print("   ✓ All certificates valid")
        print("   ✓ All required fields present")
    else:
        print("⚠️  SOME ISSUES DETECTED:")
        if stats['total'] != expected_total:
            print(f"   ❌ Incomplete grid ({stats['total']}/{expected_total})")
        if stats['errors'] > 0:
            print(f"   ❌ {stats['errors']} experiments with errors")
        if stats['invalid_certificates'] > 0:
            print(f"   ❌ {stats['invalid_certificates']} invalid certificates")
        if not coverage_complete:
            print(f"   ❌ Incomplete parameter coverage")
    
    print("="*80)
    
    return stats, issues

if __name__ == '__main__':
    results_file = Path('results_phase2_full_20251006_053336/section_a_phase2_complete.json')
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("\nSearching for results files...")
        for p in Path('.').glob('results_phase2_full_*/section_a_phase2_complete.json'):
            print(f"   Found: {p}")
            results_file = p
            break
    
    if results_file.exists():
        stats, issues = verify_phase2_results(results_file)
    else:
        print("❌ No results file found!")