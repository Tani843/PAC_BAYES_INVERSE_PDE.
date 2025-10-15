#!/usr/bin/env python3
"""
Test single experiment from phase 2 grid to verify setup and identify issues.
"""

from run_full_grid_phase2 import run_single_experiment_phase2, create_experiment_ecosystem

# Test one configuration
test_config = {
    's': 3,
    'sensor_positions': [0.25, 0.50, 0.75],
    'sigma': 0.1,
    'n_x': 50,
    'T': 0.5,
    'lambda': 1.0,
    'm': 3,
    'n_t': 50,
    'seed': 101,
    'delta': 0.05,
    'alpha': 1e-3,
    'M': 1000,
    'R': 50,
    'n': 150,
    'c': 1.0,
    'Delta_x': 1.0/50  # Spatial mesh size = 1/n_x
}

print("Starting single experiment test...")
print(f"Test config: {test_config}")

try:
    # Create the experiment ecosystem
    print("Creating experiment ecosystem...")
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = create_experiment_ecosystem()
    print("✓ Ecosystem created successfully")
    
    # Run single experiment
    print("Running single experiment...")
    result = run_single_experiment_phase2(
        test_config, DataGenerator, Prior, LossFunction, 
        Solver, GibbsPosterior, Certificate
    )
    
    print(f"✓ Test successful!")
    print(f"Result keys: {list(result.keys())}")
    
    if 'mcmc' in result:
        print(f"MCMC converged: {result['mcmc']['converged']}")
        print(f"Acceptance rate: {result['mcmc']['acceptance_rate']:.3f}")
        print(f"ESS min: {result['mcmc']['ess_min']:.1f}")
    
    if 'certificate' in result:
        print(f"Certificate valid: {result['certificate']['valid']}")
    
    if 'performance' in result:
        print(f"Runtime: {result['performance']['runtime']:.2f}s")
    
    if 'error' in result:
        print(f"❌ Error in result: {result['error']}")
        if 'traceback' in result:
            print(f"Traceback: {result['traceback']}")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()