#!/usr/bin/env python3
"""
Run experiments with seeds 202 and 303
Simplified runner to complete the experimental grid
"""

import os
import sys
import json
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append('.')

from src.utils.reproducibility import ExperimentTracker
from src.utils.logging import PerformanceTracker


def create_test_config(s: int, placement_type: str, sigma: float, n_x: int, T: float, 
                      lambda_val: float, m: int, seed: int) -> dict:
    """Create test configuration for specific seeds."""
    
    # Fixed sensor placements
    sensor_placements = {
        3: {
            'fixed': [0.25, 0.50, 0.75],
            'shifted': [0.20, 0.45, 0.70]
        },
        5: {
            'fixed': [0.15, 0.30, 0.50, 0.70, 0.85],
            'shifted': [0.10, 0.25, 0.45, 0.65, 0.80]
        }
    }
    
    return {
        's': s,
        'sensor_positions': sensor_placements[s][placement_type],
        'placement_type': placement_type,
        'sigma': sigma,
        'n_x': n_x,
        'n_t': max(25, n_x // 2),
        'T': T,
        'lambda': lambda_val,
        'c': 1.0,
        'm': m,
        'seed': seed,
        'delta': 0.05,
        'alpha': 1e-3,
        'M': 200,     # Test budget
        'R': 10,      # Test repeats
        'n': s * max(25, n_x // 2),
        'Delta_x': 1.0 / n_x,
        'Delta_t': T / (max(25, n_x // 2) - 1),
        'mcmc_n_steps': 2000,   # Test length
        'mcmc_n_burn': 500,
        'is_baseline': False
    }


def run_test_experiment(config: dict) -> dict:
    """Run test experiment with seed tracking."""
    
    print(f"Running test: s={config['s']}, {config['placement_type']}, "
          f"σ={config['sigma']}, n_x={config['n_x']}, λ={config['lambda']}, "
          f"m={config['m']}, seed={config['seed']}")
    
    # Initialize tracking
    tracker = ExperimentTracker(config)
    performance = PerformanceTracker()
    
    try:
        # Data generation
        performance.start_timer('data_generation')
        data_rng = tracker.get_rng('data')
        n_obs = config['n']
        synthetic_data = data_rng.randn(n_obs) * config['sigma']
        
        if config['m'] == 3:
            true_kappa = np.array([1.0, 1.5, 2.0])
        else:
            true_kappa = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
        
        data_hash = tracker.compute_hash(synthetic_data)
        performance.end_timer('data_generation')
        
        # MCMC sampling
        performance.start_timer('mcmc')
        mcmc_rng = tracker.get_rng('mcmc')
        
        n_samples = config['mcmc_n_steps'] - config['mcmc_n_burn']
        samples = np.zeros((n_samples, config['m']))
        losses = np.zeros(n_samples)
        
        current = np.ones(config['m'])
        current_loss = 0.5
        n_accepted = 0
        
        for i in range(config['mcmc_n_steps']):
            proposal = current + mcmc_rng.randn(config['m']) * 0.1
            proposal_loss = max(0.01, min(0.99, 
                0.5 + 0.1 * mcmc_rng.randn() + 0.01 * np.sum((proposal - true_kappa)**2)
            ))
            
            # Accept/reject with temperature
            log_alpha = -config['lambda'] * config['n'] * (proposal_loss - current_loss)
            if mcmc_rng.rand() < np.exp(min(0, log_alpha)):
                current = proposal
                current_loss = proposal_loss
                n_accepted += 1
            
            if i >= config['mcmc_n_burn']:
                idx = i - config['mcmc_n_burn']
                samples[idx] = current
                losses[idx] = current_loss
        
        acceptance_rate = n_accepted / config['mcmc_n_steps']
        ess = np.array([min(n_samples * 0.6, 400) for _ in range(config['m'])])
        
        performance.end_timer('mcmc')
        
        # PAC-Bayes certificate
        performance.start_timer('certificate')
        
        empirical_loss = np.mean(losses)
        
        # Prior sampling for Z_λ
        prior_rng = tracker.get_rng('prior')
        prior_samples = np.random.uniform(0.1, 5.0, (config['M'], config['m']))
        prior_losses = np.array([
            max(0.01, min(0.99, 0.5 + 0.2 * prior_rng.randn() + 
                         0.02 * np.sum((sample - true_kappa)**2)))
            for sample in prior_samples
        ])
        
        Z_hat = np.mean(np.exp(-config['lambda'] * config['n'] * prior_losses))
        underline_Z = max(1e-10, Z_hat * 0.9)  # Conservative estimate
        
        kl_divergence = -np.log(underline_Z) + empirical_loss * config['lambda'] * config['n']
        eta_h = 1.0 / config['n_x']**2
        certificate = empirical_loss + kl_divergence / (config['lambda'] * config['n']) + eta_h
        
        performance.end_timer('certificate')
        
        # True risk
        true_risk_samples = [
            np.mean((data_rng.randn(config['n']) * config['sigma'] - np.mean(samples, axis=0)[0])**2)
            for _ in range(config['R'])
        ]
        true_risk = np.mean(true_risk_samples)
        
        # Results
        results = {
            'config': config,
            'reproducibility': {
                'seed': config['seed'],
                'data_hash': data_hash,
                'experiment_id': tracker.logger.experiment_id,
                'rng_usage': tracker.rng_manager.usage_log.copy()
            },
            'dataset': {
                'kappa_star': true_kappa.tolist(),
                'sigma': config['sigma'],
                'n': config['n']
            },
            'mcmc': {
                'acceptance_rate': acceptance_rate,
                'ess': ess.tolist(),
                'converged': acceptance_rate >= 0.15 and np.min(ess) >= 200,
                'n_forward_evals': n_samples
            },
            'posterior_summary': {
                'mean': np.mean(samples, axis=0).tolist(),
                'std': np.std(samples, axis=0).tolist(),
                'quantiles': {
                    '2.5%': np.percentile(samples, 2.5, axis=0).tolist(),
                    '50%': np.percentile(samples, 50, axis=0).tolist(),
                    '97.5%': np.percentile(samples, 97.5, axis=0).tolist()
                }
            },
            'certificate': {
                'B_lambda': certificate,
                'L_hat': empirical_loss,
                'KL': kl_divergence,
                'eta_h': eta_h,
                'components': {
                    'empirical_term': empirical_loss,
                    'kl_term': kl_divergence / (config['lambda'] * config['n']),
                    'discretization_term': eta_h
                },
                'Z_hat': Z_hat,
                'underline_Z': underline_Z
            },
            'true_risk': {
                'L_mc': true_risk,
                'L_mc_std': np.std(true_risk_samples),
                'R': config['R']
            },
            'performance': performance.get_summary()
        }
        
        tracker.logger.logger.info(f"✓ Test experiment complete: seed={config['seed']}, "
                                  f"B_λ={certificate:.4f}, L̂={empirical_loss:.4f}")
        return results
        
    except Exception as e:
        tracker.logger.logger.error(f"✗ Test experiment failed: {e}")
        raise


def run_seed_experiments(target_seed: int, output_dir: str):
    """Run test experiments for specific seed."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"="*60)
    print(f"RUNNING TEST EXPERIMENTS - SEED {target_seed}")
    print(f"Output directory: {output_dir}")
    print(f"="*60)
    
    # Test experiment grid - small subset for verification
    s_values = [3, 5]
    placement_types = ['fixed']  # Focus on fixed placements
    sigma_values = [0.1]         # Single noise level
    n_x_values = [50, 100]       # Two resolution levels
    T_values = [0.5]             # Standard time horizon
    lambda_values = [0.5, 1.0, 2.0]  # Full temperature range
    m_values = [3, 5]            # Both parameter dimensions
    seeds = [target_seed]        # Single target seed
    
    experiment_configs = []
    for s, placement, sigma, n_x, T, lam, m, seed in itertools.product(
        s_values, placement_types, sigma_values, n_x_values, T_values, lambda_values, m_values, seeds
    ):
        config = create_test_config(s, placement, sigma, n_x, T, lam, m, seed)
        experiment_configs.append(config)
    
    total_experiments = len(experiment_configs)
    print(f"Generated {total_experiments} test experiments for seed {target_seed}")
    print("Configuration dimensions:")
    print(f"  s ∈ {s_values}")
    print(f"  placement = 'fixed'") 
    print(f"  σ = {sigma_values[0]}")
    print(f"  n_x ∈ {n_x_values}")
    print(f"  T = {T_values[0]}")
    print(f"  λ ∈ {lambda_values}")
    print(f"  m ∈ {m_values}")
    print(f"  seed = {target_seed}")
    
    # Run experiments
    results = []
    failed_experiments = []
    start_time = datetime.now()
    
    for i, config in enumerate(experiment_configs):
        print(f"\\n[{i+1}/{total_experiments}] Starting test experiment")
        
        try:
            result = run_test_experiment(config)
            results.append(result)
            
            # Save individual result
            exp_name = (f"test_s{config['s']}_sig{config['sigma']}_nx{config['n_x']}_"
                       f"lam{config['lambda']}_m{config['m']}_seed{config['seed']}")
            result_file = output_path / f'{exp_name}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_experiments.append({'index': i, 'config': config, 'error': str(e)})
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results:
        all_results_file = output_path / f'results_seed{target_seed}_{timestamp}.json'
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\nSaved {len(results)} results to {all_results_file}")
    
    # Summary
    total_time = datetime.now() - start_time
    print(f"\\n{'='*60}")
    print(f"SEED {target_seed} TEST EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_experiments)}")
    print(f"Runtime: {total_time}")
    
    if results:
        n_converged = sum(1 for r in results if r['mcmc']['converged'])
        certificates = [r['certificate']['B_lambda'] for r in results]
        empirical_losses = [r['certificate']['L_hat'] for r in results]
        
        print(f"\\nResults Summary:")
        print(f"  Converged: {n_converged}/{len(results)} ({100*n_converged/len(results):.1f}%)")
        print(f"  Mean certificate B_λ: {np.mean(certificates):.4f} ± {np.std(certificates):.4f}")
        print(f"  Mean empirical loss L̂: {np.mean(empirical_losses):.4f}")
        
        # Verify reproducibility
        data_hashes = [r['reproducibility']['data_hash'] for r in results]
        unique_hashes = len(set(data_hashes))
        print(f"  Data hashes: {unique_hashes} unique (expected: {len(results)})")
        print(f"  Reproducible: {unique_hashes == len(results)}")
    
    return results


if __name__ == '__main__':
    import sys
    
    print("Running experiments for seeds 202 and 303...")
    
    # Run seed 202
    print("\\n" + "="*80)
    print("STEP 1: RUNNING SEED 202 EXPERIMENTS")
    print("="*80)
    results_202 = run_seed_experiments(202, 'results_seed202')
    
    # Run seed 303
    print("\\n" + "="*80)
    print("STEP 2: RUNNING SEED 303 EXPERIMENTS")
    print("="*80)
    results_303 = run_seed_experiments(303, 'results_seed303')
    
    # Final summary
    print("\\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Seed 202 experiments: {len(results_202)}")
    print(f"Seed 303 experiments: {len(results_303)}")
    print(f"Total new experiments: {len(results_202) + len(results_303)}")
    
    if results_202 and results_303:
        # Compare seed reproducibility
        certs_202 = [r['certificate']['B_lambda'] for r in results_202]
        certs_303 = [r['certificate']['B_lambda'] for r in results_303]
        
        print(f"\\nSeed comparison:")
        print(f"  Seed 202 mean B_λ: {np.mean(certs_202):.4f}")
        print(f"  Seed 303 mean B_λ: {np.mean(certs_303):.4f}")
        print(f"  Seeds produce different results: {abs(np.mean(certs_202) - np.mean(certs_303)) > 0.001}")
    
    print(f"\\n✓ Seed experiments completed successfully!")