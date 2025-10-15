#!/usr/bin/env python3
"""
Classical baseline experiments with CORRECT MCMC interface
Based on successful PAC-Bayes pattern from run_full_grid_phase2_robust_final.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import signal

from config.experiment_config import ExperimentConfig
from src.forward_model.heat_equation import HeatEquationSolver
from src.data.data_generator import DataGenerator
from src.inference.loss_functions import Prior, BoundedLoss
from src.inference.gibbs_posterior import ClassicalPosterior  # Use from gibbs_posterior.py
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Classical experiment timeout")

def run_classical_baseline_correct_interface():
    """
    Classical baseline following EXACT successful PAC-Bayes pattern
    """
    
    output_dir = Path(f'classical_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("CLASSICAL BASELINE EXPERIMENTS")
    print("Using PROVEN MCMC interface from successful PAC-Bayes")
    print("=" * 60)
    
    baseline_configs = []
    config_gen = ExperimentConfig()
    
    # Simplified config set for baseline (12 experiments)
    for s in [3, 5]:
        placement = 'fixed'
        sensor_positions = config_gen.sensor_placements[s][placement]
        
        for sigma in [0.05, 0.10, 0.20]:
            for m in [3, 5]:
                for seed in [101]:  # Single seed for baseline
                    config = {
                        's': s,
                        'sensor_positions': sensor_positions,
                        'placement_type': placement,
                        'sigma': sigma,
                        'n_x': 100,
                        'n_t': 50,
                        'T': 0.5,
                        'm': m,
                        'seed': seed,
                        'n': s * 50
                    }
                    baseline_configs.append(config)
    
    print(f"Running {len(baseline_configs)} classical baseline experiments")
    results = []
    
    for i, config in enumerate(baseline_configs):
        print(f"\n[{i+1}/{len(baseline_configs)}] Classical baseline")
        print(f"  s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}, seed={config['seed']}")
        
        # Set timeout protection (same as PAC-Bayes)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15 * 60)  # 15 minute timeout
        
        try:
            # Generate data (EXACT same as PAC-Bayes)
            data_gen = DataGenerator(config)
            dataset = data_gen.generate_dataset()
            
            # Setup solver (EXACT same as PAC-Bayes)
            solver = HeatEquationSolver(
                n_x=config['n_x'],
                n_t=config['n_t'],
                T=config['T']
            )
            
            # Setup prior (EXACT same as PAC-Bayes)
            prior = Prior(
                m=config['m'],
                mu_0=0.0,
                tau2=1.0,
                kappa_min=0.1,
                kappa_max=5.0
            )
            
            # Classical posterior (use the one from gibbs_posterior.py)
            posterior = ClassicalPosterior(
                y=dataset['noisy_data'],
                solver=solver,
                prior=prior,
                sigma=config['sigma'],
                sensor_positions=dataset['sensor_positions'],
                time_grid=dataset['time_grid']
            )
            
            # MCMC sampler (EXACT same pattern as successful PAC-Bayes)
            sampler = AdaptiveMetropolisHastingsPhase2(
                posterior=posterior,
                seed=config['seed']  # Use integer seed
            )
            
            # Set Phase 2 parameters (same as PAC-Bayes)
            sampler.ess_target = 200  # Lower for baseline
            sampler.chunk_size = 2000
            sampler.max_steps = 15000
            
            # Run adaptive MCMC (EXACT same call as PAC-Bayes)
            mcmc_results = sampler.run_adaptive_length(n_burn=1000)
            
            # Cancel timeout
            signal.alarm(0)
            
            # Extract results
            samples = mcmc_results.get('samples', [])
            if len(samples) > 0:
                # Compute credible intervals
                credible_intervals = {}
                for j in range(config['m']):
                    param_samples = samples[:, j]
                    credible_intervals[f'kappa_{j}'] = {
                        'mean': float(np.mean(param_samples)),
                        'q025': float(np.percentile(param_samples, 2.5)),
                        'q975': float(np.percentile(param_samples, 97.5))
                    }
                
                result = {
                    'config': config,
                    'status': 'success',
                    'mcmc': {
                        'acceptance_rate': mcmc_results.get('overall_acceptance_rate', 0.0),
                        'min_ess': float(np.min(mcmc_results.get('final_ess', [0]))),
                        'converged': mcmc_results.get('converged', False),
                        'n_samples': len(samples)
                    },
                    'credible_intervals': credible_intervals,
                    'method': 'classical'
                }
                
                print(f"  ✓ Success: acc={result['mcmc']['acceptance_rate']:.1%}, ESS={result['mcmc']['min_ess']:.0f}")
            else:
                result = {'config': config, 'status': 'error', 'error': 'No samples generated'}
                print(f"  ✗ Error: No samples generated")
            
        except TimeoutException:
            result = {'config': config, 'status': 'timeout', 'error': 'Experiment timeout (15 min)'}
            print(f"  ⏱ Timeout after 15 minutes")
            
        except Exception as e:
            result = {'config': config, 'status': 'error', 'error': str(e)}
            print(f"  ✗ Error: {e}")
            
        finally:
            signal.alarm(0)  # Always cancel timeout
        
        results.append(result)
        
        # Save checkpoint
        if (i + 1) % 5 == 0:
            with open(output_dir / f'checkpoint_{i+1}.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    with open(output_dir / 'classical_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n" + "=" * 60)
    print(f"CLASSICAL BASELINE COMPLETE")
    print(f"Success rate: {success_count}/{len(results)} = {success_count/len(results)*100:.1f}%")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)
    
    return results

if __name__ == '__main__':
    run_classical_baseline_correct_interface()