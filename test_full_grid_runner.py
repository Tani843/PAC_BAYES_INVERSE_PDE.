#!/usr/bin/env python3
"""
Test the full Section A grid runner with a small subset
Validates the complete infrastructure before running all 1,728 experiments
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('.')

from config.experiment_config_fixed import ExperimentConfig
from run_full_section_a_grid import run_pac_bayes_experiment


def test_full_grid_runner():
    """Test the full grid runner with a representative subset."""
    
    print("="*60)
    print("TESTING FULL SECTION A GRID INFRASTRUCTURE")
    print("="*60)
    
    # Initialize configuration
    config = ExperimentConfig()
    
    # Generate complete grid to verify structure
    all_experiments = config.get_experiment_grid(include_appendix=False)
    print(f"✓ Complete grid generated: {len(all_experiments)} experiments")
    
    # Select a representative test subset (one from each major parameter combination)
    test_subset = []
    
    # Pick representative experiments covering parameter space
    test_cases = [
        # (s, placement, sigma, n_x, T, lambda, m, n_t, seed)
        (3, 'fixed', 0.05, 50, 0.3, 0.5, 3, 50, 101),      # Min values
        (5, 'shifted', 0.20, 100, 0.5, 2.0, 5, 100, 303),  # Max values  
        (3, 'shifted', 0.10, 100, 0.3, 1.0, 5, 50, 202),   # Mixed values
        (5, 'fixed', 0.05, 50, 0.5, 0.5, 3, 100, 101),     # Alternative mix
    ]
    
    # Find matching experiments in the grid
    for target in test_cases:
        s, placement, sigma, n_x, T, lambda_val, m, n_t, seed = target
        
        matching = [exp for exp in all_experiments if (
            exp['s'] == s and exp['placement_type'] == placement and 
            exp['sigma'] == sigma and exp['n_x'] == n_x and exp['T'] == T and
            exp['lambda'] == lambda_val and exp['m'] == m and exp['n_t'] == n_t and
            exp['seed'] == seed
        )]
        
        if matching:
            test_subset.append(matching[0])
            print(f"✓ Found test case: s={s}, {placement}, σ={sigma}, n_x={n_x}, T={T}, λ={lambda_val}, m={m}, n_t={n_t}, seed={seed}")
    
    print(f"\nSelected {len(test_subset)} representative test experiments")
    
    # Create test output directory
    test_output_dir = Path('test_full_grid_results')
    test_output_dir.mkdir(exist_ok=True)
    
    # Run test experiments
    print(f"\n" + "="*60)
    print("RUNNING TEST EXPERIMENTS")
    print("="*60)
    
    test_results = []
    start_time = datetime.now()
    
    for i, exp_config in enumerate(test_subset):
        print(f"\n[{i+1}/{len(test_subset)}] Testing experiment...")
        
        try:
            result = run_pac_bayes_experiment(exp_config)
            test_results.append(result)
            
            # Save individual result
            exp_file = test_output_dir / f"test_{result['experiment_id']}.json"
            with open(exp_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            print(f"✓ Success: B_λ={result['certificate']['B_lambda']:.4f}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    test_time = datetime.now() - start_time
    
    # Analysis
    print(f"\n" + "="*60)
    print("TEST RESULTS ANALYSIS")
    print("="*60)
    
    if test_results:
        certificates = [r['certificate']['B_lambda'] for r in test_results]
        empirical_losses = [r['certificate']['L_hat'] for r in test_results]
        # Get total runtime from timings
        runtimes = []
        for r in test_results:
            total_time = sum(timing['total'] for timing in r['performance']['timings'].values())
            runtimes.append(total_time)
        
        print(f"✓ All {len(test_results)} test experiments completed successfully")
        print(f"Total test time: {test_time}")
        print(f"Average per experiment: {test_time.total_seconds()/len(test_results):.1f} seconds")
        
        print(f"\nCertificate Analysis:")
        print(f"  B_λ range: [{min(certificates):.4f}, {max(certificates):.4f}]")
        print(f"  B_λ mean: {sum(certificates)/len(certificates):.4f}")
        print(f"  L̂ mean: {sum(empirical_losses)/len(empirical_losses):.4f}")
        
        print(f"\nPerformance Analysis:")
        print(f"  Runtime per experiment: {sum(runtimes)/len(runtimes):.2f}s average")
        
        # Convergence check
        converged = sum(1 for r in test_results if r['mcmc']['converged'])
        print(f"  MCMC convergence: {converged}/{len(test_results)} ({100*converged/len(test_results):.0f}%)")
        
        # Extrapolate to full grid
        avg_time_per_exp = test_time.total_seconds() / len(test_results)
        estimated_full_time_hours = (avg_time_per_exp * 1728) / 3600
        
        print(f"\n" + "="*60)
        print("FULL GRID PROJECTIONS")
        print("="*60)
        print(f"Based on test results:")
        print(f"  Average time per experiment: {avg_time_per_exp:.1f} seconds")
        print(f"  Estimated time for 1,728 experiments: {estimated_full_time_hours:.1f} hours ({estimated_full_time_hours/24:.1f} days)")
        print(f"  Expected storage: ~{1728 * 50}KB of result files")
        
        # Save test summary
        summary = {
            'test_time': str(test_time),
            'experiments_tested': len(test_results),
            'success_rate': len(test_results) / len(test_subset),
            'avg_time_per_experiment': avg_time_per_exp,
            'estimated_full_grid_hours': estimated_full_time_hours,
            'certificate_stats': {
                'mean': sum(certificates) / len(certificates),
                'min': min(certificates),
                'max': max(certificates)
            },
            'convergence_rate': converged / len(test_results)
        }
        
        summary_file = test_output_dir / 'test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Test infrastructure validation complete!")
        print(f"Test results saved to: {test_output_dir}")
        print(f"\nReady to run full 1,728 experiment grid with:")
        print(f"  python3 run_full_section_a_grid.py")
        
        return True
    
    else:
        print("✗ No test experiments completed successfully")
        return False


if __name__ == '__main__':
    print("Testing full Section A grid infrastructure...")
    
    success = test_full_grid_runner()
    
    if success:
        print("\n🎉 All systems ready for full 1,728 experiment execution!")
    else:
        print("\n❌ Infrastructure test failed. Check configuration before running full grid.")