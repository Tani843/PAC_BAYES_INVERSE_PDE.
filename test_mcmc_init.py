#!/usr/bin/env python3
"""
Test MCMC initialization to isolate hanging issue.
"""

import numpy as np
import sys
import time

# Add project root to path
sys.path.append('.')

from config.experiment_config import ExperimentConfig
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

def test_mcmc_init():
    """Test MCMC initialization."""
    
    print("Testing MCMC initialization...")
    
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
    
    # Generate data and setup posterior (we know this works)
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
    
    print("✓ Posterior ready")
    
    # Test MCMC initialization
    print("Creating MCMC sampler...")
    start_time = time.time()
    
    try:
        sampler = AdaptiveMetropolisHastingsPhase2(
            posterior=posterior,
            initial_scale=0.01,
            seed=exp_config['seed'],
            ess_target=200,     # Reduced from default
            chunk_size=5000,
            max_steps=25000,
            use_block_updates=True
        )
        
        init_time = time.time() - start_time
        print(f"✓ MCMC sampler created in {init_time:.3f}s")
        
        # Test initial sample generation
        print("Testing initial sample generation...")
        start_time = time.time()
        kappa_init = posterior.prior.sample(1, seed=exp_config['seed'])[0]
        sample_time = time.time() - start_time
        print(f"✓ Initial sample generated in {sample_time:.3f}s: {kappa_init}")
        
        # Test posterior evaluation with initial sample
        print("Testing posterior evaluation with initial sample...")
        start_time = time.time()
        log_p_init = posterior.log_posterior(kappa_init)
        eval_time = time.time() - start_time
        print(f"✓ Log posterior evaluated in {eval_time:.3f}s: {log_p_init:.3f}")
        
        print("✅ MCMC initialization successful!")
        
    except Exception as e:
        init_time = time.time() - start_time
        print(f"❌ MCMC initialization failed after {init_time:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return sampler

if __name__ == '__main__':
    try:
        sampler = test_mcmc_init()
        print("✅ MCMC initialization test passed!")
    except Exception as e:
        print(f"❌ MCMC initialization test failed: {e}")