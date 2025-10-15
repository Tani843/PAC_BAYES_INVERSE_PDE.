#!/usr/bin/env python3
"""
Simple test run with just 3 experiments to debug the issue.
"""

import json
import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
sys.path.append('.')

from config.experiment_config import ExperimentConfig

def test_simple_run():
    """Test with just first 3 experiments."""
    
    print("="*50)
    print("SIMPLE TEST RUN - 3 EXPERIMENTS")
    print("="*50)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'test_simple_{timestamp}')
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get experiments
    config = ExperimentConfig()
    all_experiments = config.get_experiment_grid(include_appendix=False)
    experiments = all_experiments[:3]  # Just first 3
    
    print(f"Testing {len(experiments)} experiments")
    
    # Import main script parts
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", "run_full_grid_phase2.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    # Create ecosystem
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = main_module.create_experiment_ecosystem()
    
    # Results tracking
    all_results = []
    
    # Run experiments
    for i, exp_config in enumerate(experiments):
        print(f"\n[{i+1:2d}/{len(experiments)}] Config: s={exp_config['s']}, "
              f"σ={exp_config['sigma']:.2f}, λ={exp_config['lambda']}, "
              f"seed={exp_config['seed']}")
        
        start_time = time.time()
        
        try:
            result = main_module.run_single_experiment_phase2(
                exp_config, DataGenerator, Prior, LossFunction, 
                Solver, GibbsPosterior, Certificate
            )
            
            all_results.append(result)
            runtime = time.time() - start_time
            
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✓ Success: acc={result['mcmc']['acceptance_rate']:.3f}, "
                      f"ESS={result['mcmc']['ess_min']:.1f}, "
                      f"valid={result['certificate']['valid']}, "
                      f"time={runtime:.1f}s")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            traceback.print_exc()
            
            result = {
                'experiment_id': f"error_{i}",
                'config': exp_config,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            all_results.append(result)
    
    # Save results
    results_file = output_dir / 'simple_test_results.json'
    safe_results = main_module.make_json_serializable(all_results)
    with open(results_file, 'w') as f:
        json.dump(safe_results, f, indent=2)
    
    print(f"\n✅ Simple test complete! Results saved to: {results_file}")
    
    return all_results

if __name__ == '__main__':
    try:
        results = test_simple_run()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()