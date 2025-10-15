#!/usr/bin/env python3
"""
Test posterior evaluation to isolate hanging issue.
"""

import numpy as np
import sys
import time

# Add project root to path
sys.path.append('.')

from config.experiment_config import ExperimentConfig

def test_posterior():
    """Test posterior evaluation."""
    
    print("Testing posterior evaluation...")
    
    # Get config and create ecosystem
    config = ExperimentConfig()
    experiments = config.get_experiment_grid(include_appendix=False)
    exp_config = experiments[0]
    
    print(f"Using config: s={exp_config['s']}, m={exp_config['m']}")
    
    # Import ecosystem from main script
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", "run_full_grid_phase2.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    print("Creating ecosystem...")
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = main_module.create_experiment_ecosystem()
    
    print("✓ Ecosystem created")
    
    # Generate data
    print("Generating dataset...")
    data_gen = DataGenerator(exp_config)
    dataset = data_gen.generate_dataset()
    print(f"✓ Dataset generated: {dataset['n_observations']} observations")
    
    # Setup components
    print("Setting up solver and loss...")
    solver = Solver(exp_config)
    loss_fn = LossFunction(c=exp_config['c'], sigma=exp_config['sigma'])
    prior = Prior(m=exp_config['m'], kappa_min=0.1, kappa_max=5.0)
    print("✓ Components ready")
    
    # Create posterior
    print("Creating posterior...")
    posterior = GibbsPosterior(
        dataset=dataset,
        solver=solver,
        loss_fn=loss_fn,
        prior=prior,
        lambda_val=exp_config['lambda'],
        config=exp_config
    )
    print("✓ Posterior created")
    
    # Test posterior evaluation
    print("Testing posterior evaluation...")
    test_kappa = dataset['kappa_star']
    print(f"Test kappa: {test_kappa}")
    
    start_time = time.time()
    try:
        print("  Computing log posterior...")
        log_p = posterior.log_posterior(test_kappa)
        runtime = time.time() - start_time
        print(f"✓ Log posterior: {log_p:.3f} (computed in {runtime:.3f}s)")
        
        # Test a few more evaluations
        for i in range(3):
            test_kappa_rand = prior.sample(1, seed=100+i)[0]
            start_time = time.time()
            log_p_rand = posterior.log_posterior(test_kappa_rand)
            runtime = time.time() - start_time
            print(f"  Random kappa {i+1}: log_p = {log_p_rand:.3f} ({runtime:.3f}s)")
            
    except Exception as e:
        runtime = time.time() - start_time
        print(f"❌ Posterior evaluation failed after {runtime:.3f}s: {e}")
        raise
    
    print("✅ Posterior evaluation successful!")
    return True

if __name__ == '__main__':
    try:
        test_posterior()
        print("✅ Posterior test passed!")
    except Exception as e:
        print(f"❌ Posterior test failed: {e}")
        import traceback
        traceback.print_exc()