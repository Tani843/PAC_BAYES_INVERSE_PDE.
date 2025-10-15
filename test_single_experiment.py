#!/usr/bin/env python3
"""
Test single experiment to debug the hanging issue.
"""
import json
import numpy as np
import sys
import time
import traceback

# Add project root to path
sys.path.append('.')

# Import configuration
from config.experiment_config import ExperimentConfig

# Import Phase 2 MCMC
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def test_single_experiment():
    """Test running a single experiment."""
    
    print("Loading configuration...")
    config = ExperimentConfig()
    experiments = config.get_experiment_grid(include_appendix=False)
    
    print(f"Loaded {len(experiments)} experiments")
    
    # Take first experiment
    exp_config = experiments[0]
    print(f"Testing config: {exp_config}")
    
    print("Creating ecosystem...")
    
    # Import the ecosystem creation function from the main script
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", "run_full_grid_phase2.py")
    main_module = importlib.util.module_from_spec(spec)
    
    print("Loading main script...")
    spec.loader.exec_module(main_module)
    
    print("Creating ecosystem...")
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = main_module.create_experiment_ecosystem()
    
    print("✓ Ecosystem created successfully")
    
    return True

if __name__ == '__main__':
    try:
        result = test_single_experiment()
        print("✓ Test completed successfully")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()