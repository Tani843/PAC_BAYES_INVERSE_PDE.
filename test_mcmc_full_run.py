#!/usr/bin/env python3
"""
Test full MCMC run with actual parameters to isolate hanging issue.
"""

import numpy as np
import sys
import time

# Add project root to path
sys.path.append('.')

from config.experiment_config import ExperimentConfig
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

def test_mcmc_full_run():
    """Test full MCMC run with realistic parameters."""
    
    print("Testing full MCMC run...")
    
    # Get config and create ecosystem
    config = ExperimentConfig()
    experiments = config.get_experiment_grid(include_appendix=False)
    exp_config = experiments[0]
    
    # Import ecosystem from main script
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", "run_full_grid_phase2.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    # Create ecosystem
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = main_module.create_experiment_ecosystem()
    
    # Generate data and setup posterior
    data_gen = DataGenerator(exp_config)
    dataset = data_gen.generate_dataset()
    solver = Solver(exp_config)
    loss_fn = LossFunction(c=exp_config['c'], sigma=exp_config['sigma'])
    prior = Prior(m=exp_config['m'], kappa_min=0.1, kappa_max=5.0)
    
    posterior = GibbsPosterior(
        dataset=dataset,
        solver=solver,
        loss_fn=loss_fn,
        prior=prior,
        lambda_val=exp_config['lambda'],
        config=exp_config
    )
    
    # Create MCMC sampler with actual parameters from script
    sampler = AdaptiveMetropolisHastingsPhase2(
        posterior=posterior,
        initial_scale=0.01,
        seed=exp_config['seed'],
        ess_target=200,     # From script
        chunk_size=5000,    # From script  
        max_steps=25000,    # From script
        use_block_updates=True
    )
    
    print("✓ MCMC sampler ready")
    print("Starting run_adaptive_length with actual parameters...")
    print("  n_burn=2000 (as in script)")
    print("  ess_target=200")
    print("  chunk_size=5000") 
    print("  max_steps=25000")
    
    start_time = time.time()
    try:
        # This is the exact call from the script
        mcmc_results = sampler.run_adaptive_length(n_burn=2000)
        
        runtime = time.time() - start_time
        
        print(f"✅ MCMC run completed in {runtime:.1f}s")
        print(f"  Samples: {len(mcmc_results['samples'])}")
        print(f"  ESS min: {np.min(mcmc_results['final_ess']):.1f}")
        print(f"  ESS mean: {np.mean(mcmc_results['final_ess']):.1f}")
        print(f"  Acceptance rate: {mcmc_results.get('overall_acceptance_rate', 'N/A'):.3f}")
        print(f"  Converged: {mcmc_results['converged']}")
        print(f"  Total steps: {mcmc_results['total_steps']}")
        
        return mcmc_results
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"❌ MCMC run failed after {runtime:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    try:
        results = test_mcmc_full_run()
        print("✅ Full MCMC test passed!")
    except Exception as e:
        print(f"❌ Full MCMC test failed: {e}")