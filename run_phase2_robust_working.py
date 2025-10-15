#!/usr/bin/env python3
"""
Robust Phase 2 runner - WORKING VERSION
Uses proven ecosystem approach with timeout protection
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import signal
import argparse
from typing import Dict, List, Optional
import time
import traceback
import sys

# Add project root to path
sys.path.append('.')

from config.experiment_config import ExperimentConfig

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Experiment timed out")

def run_single_experiment_with_timeout(config_dict: Dict, 
                                      timeout_sec: int = 900,
                                      max_steps_hard: int = 25000) -> Dict:
    """
    Run single experiment with timeout protection using proven ecosystem.
    """
    experiment_id = f"s{config_dict['s']}_sigma{config_dict['sigma']}_nx{config_dict['n_x']}_T{config_dict['T']}_lam{config_dict['lambda']}_m{config_dict['m']}_seed{config_dict['seed']}"
    start_time = time.time()
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    try:
        print(f"  Running: s={config_dict['s']}, σ={config_dict['sigma']:.2f}, "
              f"λ={config_dict['lambda']}, seed={config_dict['seed']}")
        
        # Import the working ecosystem from the proven script
        from run_full_grid_phase2 import run_single_experiment_phase2, create_experiment_ecosystem
        
        # Create ecosystem classes
        DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = create_experiment_ecosystem()
        
        # Run the proven single experiment function with timeout protection
        result = run_single_experiment_phase2(
            config_dict, DataGenerator, Prior, LossFunction, 
            Solver, GibbsPosterior, Certificate
        )
        
        # Cancel timeout
        signal.alarm(0)
        
        # Check if the result indicates success or error
        if 'error' in result:
            return {
                'experiment_id': experiment_id,
                'config': config_dict,
                'status': 'error',
                'error': result['error'],
                'traceback': result.get('traceback', ''),
                'performance': result.get('performance', {'runtime': time.time() - start_time})
            }
        else:
            return {
                'experiment_id': experiment_id,
                'config': config_dict,
                'status': 'success',
                'dataset': result.get('dataset', {}),
                'mcmc': result.get('mcmc', {}),
                'certificate': result.get('certificate', {}),
                'posterior_summary': result.get('posterior_summary', {}),
                'performance': result.get('performance', {'runtime': time.time() - start_time})
            }
        
    except TimeoutException:
        signal.alarm(0)
        runtime = time.time() - start_time
        print(f"    TIMEOUT after {timeout_sec}s")
        return {
            'experiment_id': experiment_id,
            'config': config_dict,
            'status': 'timeout',
            'error': f'Exceeded {timeout_sec}s timeout',
            'performance': {'runtime': runtime}
        }
    except Exception as e:
        signal.alarm(0)
        runtime = time.time() - start_time
        print(f"    ERROR: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'config': config_dict,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'performance': {'runtime': runtime}
        }

def run_batch(experiments: List[Dict], 
              batch_idx: int,
              output_dir: Path,
              start_idx: int,
              batch_size: int,
              timeout_sec: int = 900,
              max_steps_hard: int = 25000) -> Dict:
    """
    Run a batch of experiments with checkpointing.
    """
    batch_results = []
    batch_summary = {
        'batch_idx': batch_idx,
        'total': len(experiments),
        'success': 0,
        'timeout': 0,
        'error': 0,
        'start_time': datetime.now().isoformat()
    }
    
    for i, exp in enumerate(experiments):
        print(f"\n[Batch {batch_idx}, Exp {i+1}/{len(experiments)}]")
        result = run_single_experiment_with_timeout(exp, timeout_sec, max_steps_hard)
        
        batch_results.append(result)
        batch_summary[result['status']] = batch_summary.get(result['status'], 0) + 1
        
        # Save individual result immediately (with numpy-safe JSON)
        exp_idx = start_idx + batch_idx * batch_size + i
        exp_file = output_dir / f"exp_{exp_idx:04d}_seed{exp['seed']}_s{exp['s']}_sig{exp['sigma']:.3f}_lam{exp['lambda']:.1f}_T{exp['T']:.1f}_m{exp['m']}.json"
        
        def convert_numpy(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert result to be JSON-serializable
        def make_json_safe(d):
            if isinstance(d, dict):
                return {k: make_json_safe(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [make_json_safe(v) for v in d]
            else:
                return convert_numpy(d)
        
        safe_result = make_json_safe(result)
        
        with open(exp_file, 'w') as f:
            json.dump(safe_result, f, indent=2)
    
    batch_summary['end_time'] = datetime.now().isoformat()
    
    # Save batch checkpoint (with numpy-safe JSON)
    def convert_numpy(obj):
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
        if isinstance(d, dict):
            return {k: make_json_safe(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [make_json_safe(v) for v in d]
        else:
            return convert_numpy(d)
    
    safe_batch_results = make_json_safe(batch_results)
    
    checkpoint_file = output_dir / f"checkpoint_batch_{batch_idx:04d}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(safe_batch_results, f, indent=2)
    
    # Save batch summary
    summary_file = output_dir / f"batch_summary_{batch_idx:04d}.json"
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    return batch_summary

def main():
    parser = argparse.ArgumentParser(description='Robust Phase 2 Grid Runner - WORKING')
    parser.add_argument('--start_idx', type=int, default=50, help='Start index (resume from)')
    parser.add_argument('--end_idx', type=int, default=1728, help='End index')
    parser.add_argument('--batch_size', type=int, default=25, help='Experiments per batch')
    parser.add_argument('--timeout_sec', type=int, default=900, help='Timeout per experiment (15 min)')
    parser.add_argument('--max_steps_hard', type=int, default=25000, help='Hard limit on MCMC steps')
    parser.add_argument('--output_dir', type=str, default='results_phase2_robust_working', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"{args.output_dir}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Get full experiment grid
    config = ExperimentConfig()
    all_experiments = config.get_experiment_grid(include_appendix=False)
    
    # Slice to requested range
    experiments_to_run = all_experiments[args.start_idx:args.end_idx]
    
    print(f"\n{'='*60}")
    print(f"ROBUST PHASE 2 GRID RUNNER - WORKING VERSION")
    print(f"{'='*60}")
    print(f"Running experiments {args.start_idx} to {args.end_idx}")
    print(f"Total: {len(experiments_to_run)} experiments")
    print(f"Batch size: {args.batch_size}")
    print(f"Timeout: {args.timeout_sec}s per experiment")
    print(f"Output: {output_dir}")
    
    # Global status tracking
    global_status = {
        'total_experiments': len(experiments_to_run),
        'start_idx': args.start_idx,
        'end_idx': args.end_idx,
        'batch_size': args.batch_size,
        'timeout_sec': args.timeout_sec,
        'max_steps_hard': args.max_steps_hard,
        'start_time': datetime.now().isoformat(),
        'batches': []
    }
    
    # Process in batches
    for batch_idx in range(0, len(experiments_to_run), args.batch_size):
        batch_end = min(batch_idx + args.batch_size, len(experiments_to_run))
        batch = experiments_to_run[batch_idx:batch_end]
        
        actual_batch_idx = batch_idx // args.batch_size
        
        print(f"\n{'='*60}")
        print(f"Processing batch {actual_batch_idx + 1}: "
              f"experiments {args.start_idx + batch_idx} to {args.start_idx + batch_end - 1}")
        
        batch_summary = run_batch(
            batch, 
            actual_batch_idx,
            output_dir,
            args.start_idx,
            args.batch_size,
            args.timeout_sec,
            args.max_steps_hard
        )
        
        global_status['batches'].append(batch_summary)
        
        # Update global status file
        status_file = output_dir / 'run_status.json'
        with open(status_file, 'w') as f:
            json.dump(global_status, f, indent=2)
        
        print(f"Batch complete: {batch_summary['success']} success, "
              f"{batch_summary['timeout']} timeout, {batch_summary['error']} error")
    
    global_status['end_time'] = datetime.now().isoformat()
    
    # Final status save
    status_file = output_dir / 'run_status.json'
    with open(status_file, 'w') as f:
        json.dump(global_status, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL BATCHES COMPLETE")
    print(f"Results in: {output_dir}")
    
    # Generate final summary
    total_success = sum(b['success'] for b in global_status['batches'])
    total_timeout = sum(b['timeout'] for b in global_status['batches'])
    total_error = sum(b['error'] for b in global_status['batches'])
    
    print(f"Final Summary:")
    print(f"  Success: {total_success}/{len(experiments_to_run)} ({total_success/len(experiments_to_run):.1%})")
    print(f"  Timeout: {total_timeout}/{len(experiments_to_run)} ({total_timeout/len(experiments_to_run):.1%})")
    print(f"  Error: {total_error}/{len(experiments_to_run)} ({total_error/len(experiments_to_run):.1%})")

if __name__ == '__main__':
    main()