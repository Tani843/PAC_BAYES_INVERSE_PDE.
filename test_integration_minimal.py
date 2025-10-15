#!/usr/bin/env python3
"""
Minimal integration test for Phase 2 full grid
"""

import sys
sys.path.append('.')

def minimal_integration_test():
    """Minimal test to verify all components load and work together."""
    
    print("=" * 50)
    print("MINIMAL PHASE 2 INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Import all components
    print("1. Testing imports...")
    try:
        from run_full_grid_phase2 import create_experiment_ecosystem, run_single_experiment_phase2
        from config.experiment_config import ExperimentConfig
        from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2
        print("  ✓ All imports successful")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False
    
    # Test 2: Create ecosystem
    print("\n2. Testing ecosystem creation...")
    try:
        DataGen, Prior, Loss, Solver, Posterior, Cert = create_experiment_ecosystem()
        print("  ✓ Ecosystem created successfully")
    except Exception as e:
        print(f"  ❌ Ecosystem creation failed: {e}")
        return False
    
    # Test 3: Get experiment config
    print("\n3. Testing experiment configuration...")
    try:
        config = ExperimentConfig()
        experiments = config.get_experiment_grid(include_appendix=False)
        test_config = experiments[0]  # First experiment
        print(f"  ✓ Got {len(experiments)} experiments, testing first one")
        print(f"    Config: s={test_config['s']}, σ={test_config['sigma']}")
    except Exception as e:
        print(f"  ❌ Config failed: {e}")
        return False
    
    # Test 4: Create components
    print("\n4. Testing component creation...")
    try:
        # Data generation
        data_gen = DataGen(test_config)
        dataset = data_gen.generate_dataset()
        print(f"  ✓ Dataset: {dataset['n_observations']} observations")
        
        # Solver and loss
        solver = Solver(test_config)
        loss_fn = Loss(c=1.0, sigma=test_config['sigma'])
        prior = Prior(m=test_config['m'])
        print("  ✓ Solver, loss, and prior created")
        
        # Posterior
        posterior = Posterior(
            dataset=dataset, solver=solver, loss_fn=loss_fn, 
            prior=prior, lambda_val=test_config['lambda'], config=test_config
        )
        print("  ✓ Posterior created")
        
        # Test posterior evaluation
        test_kappa = dataset['kappa_star']
        logp = posterior.log_posterior(test_kappa)
        print(f"  ✓ Posterior evaluation: {logp:.3f}")
        
    except Exception as e:
        print(f"  ❌ Component creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Quick MCMC test
    print("\n5. Testing Phase 2 MCMC...")
    try:
        sampler = AdaptiveMetropolisHastingsPhase2(
            posterior=posterior,
            seed=42,
            ess_target=50,    # Very low for quick test
            chunk_size=500,   # Very small
            max_steps=2000,   # Very limited
        )
        
        result = sampler.run_adaptive_length(n_burn=200)
        
        print(f"  ✓ MCMC completed:")
        print(f"    Samples: {len(result['samples'])}")
        print(f"    ESS min: {result['final_ess'].min():.1f}")
        print(f"    Acceptance: {result.get('overall_acceptance_rate', 0):.3f}")
        print(f"    Converged: {result['converged']}")
        
    except Exception as e:
        print(f"  ❌ MCMC failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n" + "="*50)
    print("✅ MINIMAL INTEGRATION TEST PASSED!")
    print("All components work together correctly.")
    print("Ready for full grid execution.")
    print("="*50)
    
    return True

if __name__ == '__main__':
    success = minimal_integration_test()
    if success:
        print(f"\n🚀 Execute full grid with: python3 run_full_grid_phase2.py")
    else:
        print(f"\n❌ Fix integration issues before proceeding")
    
    sys.exit(0 if success else 1)