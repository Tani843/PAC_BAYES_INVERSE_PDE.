#!/usr/bin/env python3
"""
Simple test of Phase 2 pipeline with just a few experiments
"""

import json
import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime

print("Starting Phase 2 simple test...")

# Add project root to path
sys.path.append('.')

print("Importing configuration...")
from config.experiment_config import ExperimentConfig

print("Importing MCMC...")
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

print("Importing main script functions...")
import importlib.util
spec = importlib.util.spec_from_file_location("main_script", "run_full_grid_phase2.py")
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

print("Creating ecosystem...")
DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = main_module.create_experiment_ecosystem()

print("Getting experiment configs...")
config = ExperimentConfig()
experiments = config.get_experiment_grid(include_appendix=False)
print(f"Found {len(experiments)} total experiments")

# Run just first 3 experiments
test_experiments = experiments[:3]
print(f"Testing with {len(test_experiments)} experiments")

print("Starting experiment loop...")
for i, exp_config in enumerate(test_experiments):
    print(f"\n[{i+1}/{len(test_experiments)}] Running experiment {i+1}")
    print(f"  Config: s={exp_config['s']}, σ={exp_config['sigma']:.2f}, λ={exp_config['lambda']}")
    
    start_time = time.time()
    try:
        result = main_module.run_single_experiment_phase2(
            exp_config, DataGenerator, Prior, LossFunction, 
            Solver, GibbsPosterior, Certificate
        )
        runtime = time.time() - start_time
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✅ Success: time={runtime:.1f}s, "
                  f"acc={result['mcmc']['acceptance_rate']:.3f}, "
                  f"ESS={result['mcmc']['ess_min']:.1f}")
    except Exception as e:
        runtime = time.time() - start_time
        print(f"  ❌ Exception after {runtime:.1f}s: {e}")

print("✅ Phase 2 simple test complete!")