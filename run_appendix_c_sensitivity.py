# run_appendix_c_sensitivity.py
"""
Appendix A.1: Sensitivity analysis for loss scale constant c.
Tests c ∈ {0.5, 1.0, 2.0} with reduced parameter grid.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def get_appendix_c_experiments():
    """Generate experiment configurations for c-sensitivity analysis."""
    experiments = []
    
    # Fixed display configuration (same as Fig. 1)
    fixed_params = {
        'n_x': 100,
        'n_t': 50,
        'T': 0.5,
        'lambda': 1.0,
        'placement_type': 'fixed',
        'delta': 0.05,
        'alpha': 1e-3,
        'M': 2000,
        'R': 100
    }
    
    # Parameter sweep
    s_values = [3, 5]
    sigma_values = [0.05, 0.10, 0.20]
    m_values = [3, 5]
    c_values = [0.5, 1.0, 2.0]
    seeds = [101, 202, 303]
    
    exp_id = 0
    for s in s_values:
        for sigma in sigma_values:
            for m in m_values:
                for c in c_values:
                    for seed in seeds:
                        config = {
                            **fixed_params,
                            's': s,
                            'sigma': sigma,
                            'm': m,
                            'c': c,
                            'seed': seed,
                            'n': s * fixed_params['n_t'],
                            'sensor_positions': get_sensor_positions(s, 'fixed'),
                            'Delta_x': 1.0 / fixed_params['n_x'],
                            'Delta_t': fixed_params['T'] / (fixed_params['n_t'] - 1),
                            'mcmc_n_steps': 10000,
                            'mcmc_n_burn': 2000,
                            'exp_id': exp_id
                        }
                        experiments.append(config)
                        exp_id += 1
    
    print(f"Total appendix experiments: {len(experiments)}")
    return experiments

def get_sensor_positions(s, placement_type='fixed'):
    """Get sensor positions matching main experiments."""
    if placement_type == 'fixed':
        if s == 3:
            return [0.25, 0.50, 0.75]
        elif s == 5:
            return [0.10, 0.30, 0.50, 0.70, 0.90]
    return []

def run_single_c_experiment(config: Dict, output_dir: Path):
    """Run single experiment with specific c value (no timeout wrapper available)."""
    from run_full_grid_phase2 import run_single_experiment_phase2, create_experiment_ecosystem

    # Create ecosystem components (no parameters needed)
    components = create_experiment_ecosystem()
    
    # Run experiment
    result = run_single_experiment_phase2(config, *components)


    filename = (f"s{config['s']}_sigma{config['sigma']:.3f}_"
                f"m{config['m']}_c{config['c']:.1f}_seed{config['seed']}.json")
    result_path = output_dir / filename
    with open(result_path, 'w') as f:
        json.dump(make_json_serializable(result), f, indent=2)

    return result

def run_appendix_c_grid():
    """Run complete c-sensitivity grid."""
    # Create output directory
    output_dir = Path('results_appendix_c')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)
    
    # Get experiments
    experiments = get_appendix_c_experiments()
    
    # Run experiments
    results = []
    for i, config in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running c={config['c']:.1f} experiment")
        print(f"  s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}, seed={config['seed']}")
        
        try:
            result = run_single_c_experiment(config, output_dir)
            results.append(result)
            print(f"  ✓ Success: L̂={result['certificate']['L_hat']:.4f}, "
                  f"B_λ={result['certificate']['B_lambda']:.4f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Save consolidated results
    consolidated_path = output_dir / 'appendix_c_all_results.json'
    with open(consolidated_path, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    
    print(f"\n✓ Completed {len(results)}/{len(experiments)} experiments")
    print(f"✓ Results saved to: {output_dir}")
    
    return results

if __name__ == '__main__':
    results = run_appendix_c_grid()