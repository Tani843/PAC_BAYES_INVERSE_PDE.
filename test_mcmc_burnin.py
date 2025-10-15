#!/usr/bin/env python3
"""
Test MCMC burn-in to isolate hanging issue.
"""

import numpy as np
import sys
import time

# Add project root to path
sys.path.append('.')

from config.experiment_config import ExperimentConfig
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

def test_mcmc_burnin():
    """Test MCMC burn-in phase."""
    
    print("Testing MCMC burn-in...")
    
    # Get config and create ecosystem (reuse working setup)
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
    
    # Create MCMC sampler
    sampler = AdaptiveMetropolisHastingsPhase2(
        posterior=posterior,
        initial_scale=0.01,
        seed=exp_config['seed'],
        ess_target=200,
        chunk_size=5000,
        max_steps=25000,
        use_block_updates=True
    )
    
    print("✓ MCMC sampler ready")
    
    # Test burn-in with very short chain
    print("Testing short burn-in (100 steps)...")
    kappa_init = posterior.prior.sample(1, seed=exp_config['seed'])[0]
    print(f"Initial kappa: {kappa_init}")
    
    start_time = time.time()
    try:
        # Try very short burn-in first
        burn_chain, kappa_final = sampler.adaptive_burn_in(kappa_init, n_burn=100)
        burnin_time = time.time() - start_time
        
        print(f"✓ Burn-in completed in {burnin_time:.3f}s")
        print(f"  Chain length: {len(burn_chain)}")
        print(f"  Final kappa: {kappa_final}")
        
        if len(burn_chain) > 0:
            print(f"  First sample: {burn_chain[0]}")
            print(f"  Last sample: {burn_chain[-1]}")
        
        print("✅ Burn-in test successful!")
        
    except Exception as e:
        burnin_time = time.time() - start_time
        print(f"❌ Burn-in failed after {burnin_time:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return sampler, burn_chain, kappa_final

if __name__ == '__main__':
    try:
        sampler, chain, final_kappa = test_mcmc_burnin()
        print("✅ MCMC burn-in test passed!")
    except Exception as e:
        print(f"❌ MCMC burn-in test failed: {e}")