#!/usr/bin/env python3
"""
Run missing experiments 0-81 with the robust Phase 2 system
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path  
sys.path.append('.')

from config.experiment_config import ExperimentConfig
from run_phase2_robust_working import run_single_experiment_with_timeout

def convert_numpy(obj):
    """Convert numpy types to JSON serializable types"""
    import numpy as np
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def make_json_safe(d):
    """Recursively convert numpy types to JSON serializable types"""
    if isinstance(d, dict):
        return {k: make_json_safe(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [make_json_safe(v) for v in d]
    else:
        return convert_numpy(d)

def run_missing_experiments():
    """Run experiments 0-81 with the robust Phase 2 system"""
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'missing_experiments_{timestamp}')
    output_dir.mkdir(exist_ok=True)
    
    print(f"=== MISSING EXPERIMENTS RUNNER ===")
    print(f"Output directory: {output_dir}")
    
    # Get experiment configurations
    config = ExperimentConfig()
    all_experiments = config.get_experiment_grid(include_appendix=False)
    
    # Get only experiments 0-81
    missing_experiments = all_experiments[0:82]
    
    print(f"Running {len(missing_experiments)} missing experiments (0-81)")
    print(f"Using robust timeout protection (15 minutes per experiment)")
    
    results = []
    start_time = datetime.now()
    
    for i in range(82):
        exp_config = missing_experiments[i]
        
        print(f"\n[Experiment {i:02d}/81] Starting...")
        print(f"  Config: s={exp_config['s']}, σ={exp_config['sigma']:.2f}, "
              f"λ={exp_config['lambda']}, T={exp_config['T']}, m={exp_config['m']}, seed={exp_config['seed']}")
        
        # Run with robust timeout protection
        result = run_single_experiment_with_timeout(
            exp_config, 
            timeout_sec=900,  # 15 minutes
            max_steps_hard=15000
        )
        results.append(result)
        
        # Make result JSON-safe and save individual experiment
        safe_result = make_json_safe(result)
        exp_file = output_dir / f'exp_{i:04d}_seed{exp_config["seed"]}_s{exp_config["s"]}_' \
                                f'sig{exp_config["sigma"]:.3f}_lam{exp_config["lambda"]:.1f}_' \
                                f'T{exp_config["T"]:.1f}_m{exp_config["m"]}.json'
        
        with open(exp_file, 'w') as f:
            json.dump(safe_result, f, indent=2)
        
        status = result.get('status', 'unknown')
        runtime = result.get('performance', {}).get('runtime', 0)
        print(f"  Result: {status} (runtime: {runtime:.1f}s)")
        print(f"  Saved: {exp_file.name}")
        
        # Save checkpoint every 10 experiments
        if (i + 1) % 10 == 0:
            checkpoint_file = output_dir / f'checkpoint_{i+1:02d}.json'
            safe_results = make_json_safe(results)
            with open(checkpoint_file, 'w') as f:
                json.dump(safe_results, f, indent=2)
            
            elapsed = datetime.now() - start_time
            success_count = sum(1 for r in results if r.get('status') == 'success')
            print(f"  Checkpoint: {i+1}/82 experiments ({success_count} successful)")
            print(f"  Elapsed time: {elapsed}")
    
    # Save final results
    safe_results = make_json_safe(results)
    final_file = output_dir / 'missing_experiments_complete.json'
    with open(final_file, 'w') as f:
        json.dump(safe_results, f, indent=2)
    
    # Final summary
    total_time = datetime.now() - start_time
    success_count = sum(1 for r in results if r.get('status') == 'success')
    timeout_count = sum(1 for r in results if r.get('status') == 'timeout')
    error_count = sum(1 for r in results if r.get('status') == 'error')
    
    print(f"\n{'='*50}")
    print(f"✅ MISSING EXPERIMENTS COMPLETE!")
    print(f"{'='*50}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Timeouts: {timeout_count} ({timeout_count/len(results)*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"Total runtime: {total_time}")
    print(f"Average per experiment: {total_time.total_seconds()/len(results):.1f} seconds")
    print(f"Results directory: {output_dir}")
    print(f"Final results file: {final_file}")
    
    return results

if __name__ == '__main__':
    run_missing_experiments()