#!/usr/bin/env python3
"""
Test the full grid Phase 2 integration on a small subset
"""

import sys
sys.path.append('.')

from run_full_grid_phase2 import create_experiment_ecosystem, run_single_experiment_phase2
from config.experiment_config import ExperimentConfig

def test_phase2_integration():
    """Test Phase 2 integration on a few representative experiments."""
    
    print("=" * 60)
    print("TESTING PHASE 2 FULL GRID INTEGRATION")
    print("=" * 60)
    
    # Get a small subset of experiments for testing
    config = ExperimentConfig()
    all_experiments = config.get_experiment_grid(include_appendix=False)
    
    # Test configurations (representative subset)
    test_indices = [0, 100, 500, 1000, 1500]  # Spread across the grid
    test_experiments = [all_experiments[i] for i in test_indices if i < len(all_experiments)]
    
    print(f"Testing {len(test_experiments)} representative experiments...")
    
    # Create ecosystem
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = create_experiment_ecosystem()
    
    results = []
    
    for i, exp_config in enumerate(test_experiments):
        print(f"\n[{i+1}/{len(test_experiments)}] Testing config:")
        print(f"  s={exp_config['s']}, σ={exp_config['sigma']:.2f}, "
              f"n_x={exp_config['n_x']}, T={exp_config['T']}, "
              f"λ={exp_config['lambda']}, m={exp_config['m']}, "
              f"seed={exp_config['seed']}")
        
        # Modify config for faster testing
        test_config = exp_config.copy()
        
        try:
            result = run_single_experiment_phase2(
                test_config, DataGenerator, Prior, LossFunction, 
                Solver, GibbsPosterior, Certificate
            )
            
            results.append(result)
            
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                mcmc = result['mcmc']
                cert = result['certificate']
                perf = result['performance']
                
                print(f"  ✓ Results:")
                print(f"    MCMC: acc={mcmc['acceptance_rate']:.3f}, "
                      f"ESS={mcmc['ess_min']:.1f}, conv={mcmc['converged']}")
                print(f"    Cert: B_λ={cert['B_lambda']:.4f}, "
                      f"L̂={cert['L_hat']:.4f}, valid={cert['valid']}")
                print(f"    Time: {perf['runtime']:.1f}s")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyze results
    successful = [r for r in results if 'error' not in r]
    
    print(f"\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    if successful:
        converged = sum(1 for r in successful if r['mcmc']['converged'])
        valid_certs = sum(1 for r in successful if r['certificate']['valid'])
        
        acceptance_rates = [r['mcmc']['acceptance_rate'] for r in successful]
        ess_values = [r['mcmc']['ess_min'] for r in successful]
        runtimes = [r['performance']['runtime'] for r in successful]
        
        print(f"✅ Success rate: {len(successful)}/{len(test_experiments)} ({len(successful)/len(test_experiments):.1%})")
        print(f"✅ Convergence: {converged}/{len(successful)} ({converged/len(successful):.1%})")
        print(f"✅ Valid certificates: {valid_certs}/{len(successful)} ({valid_certs/len(successful):.1%})")
        print(f"📊 Acceptance: {min(acceptance_rates):.3f} - {max(acceptance_rates):.3f} (avg: {sum(acceptance_rates)/len(acceptance_rates):.3f})")
        print(f"📊 Min ESS: {min(ess_values):.1f} - {max(ess_values):.1f} (avg: {sum(ess_values)/len(ess_values):.1f})")
        print(f"⏱️ Runtime: {min(runtimes):.1f}s - {max(runtimes):.1f}s (avg: {sum(runtimes)/len(runtimes):.1f}s)")
        
        # Estimated full grid time
        avg_runtime = sum(runtimes) / len(runtimes)
        estimated_total = avg_runtime * 1728 / 3600  # Convert to hours
        print(f"⏱️ Estimated full grid time: {estimated_total:.1f} hours")
        
        # Success criteria
        min_success = len(successful) >= len(test_experiments) * 0.8
        min_convergence = converged >= len(successful) * 0.5
        min_validity = valid_certs >= len(successful) * 0.3
        reasonable_time = avg_runtime < 300  # Less than 5 minutes per experiment
        
        print(f"\n🎯 READINESS ASSESSMENT:")
        print(f"   Success rate ≥ 80%: {'✓' if min_success else '✗'}")
        print(f"   Convergence ≥ 50%: {'✓' if min_convergence else '✗'}")
        print(f"   Validity ≥ 30%: {'✓' if min_validity else '✗'}")
        print(f"   Runtime ≤ 5min: {'✓' if reasonable_time else '✗'}")
        
        ready = min_success and min_convergence and min_validity and reasonable_time
        
        if ready:
            print(f"\n🚀 READY FOR FULL GRID EXECUTION")
            print(f"   Phase 2 integration is working correctly")
            print(f"   Estimated completion time: {estimated_total:.1f} hours")
        else:
            print(f"\n⚠️ NEEDS TUNING BEFORE FULL GRID")
            print(f"   Some criteria not met - consider parameter adjustments")
        
        return ready
        
    else:
        print(f"❌ NO SUCCESSFUL EXPERIMENTS")
        print(f"   Integration has serious issues")
        return False

def main():
    """Run the integration test."""
    
    ready = test_phase2_integration()
    
    if ready:
        print(f"\n" + "="*60)
        print(f"🎉 INTEGRATION TEST PASSED!")
        print(f"Ready to execute: python3 run_full_grid_phase2.py")
        print(f"="*60)
    else:
        print(f"\n" + "="*60)
        print(f"❌ INTEGRATION TEST FAILED")
        print(f"Fix issues before running full grid")
        print(f"="*60)
    
    return ready

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)