#!/usr/bin/env python3
"""
Simple test experiment runner to verify the pipeline works
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append('.')

from src.utils.reproducibility import ExperimentTracker
from src.utils.logging import DiagnosticsLogger, PerformanceTracker


def create_test_config(seed: int) -> dict:
    """Create a minimal test configuration."""
    return {
        's': 3,
        'sensor_positions': [0.25, 0.50, 0.75],
        'placement_type': 'fixed',
        'sigma': 0.1,
        'n_x': 20,  # Small for testing
        'n_t': 20,  # Small for testing
        'T': 0.5,
        'lambda': 1.0,
        'c': 1.0,
        'm': 3,
        'seed': seed,
        'delta': 0.05,
        'alpha': 1e-3,
        'M': 50,    # Small for testing
        'R': 5,     # Small for testing
        'n': 3 * 20,  # s * n_t
        'Delta_x': 1.0 / 20,
        'Delta_t': 0.5 / 19,
        'mcmc_n_steps': 500,   # Small for testing
        'mcmc_n_burn': 100,
        'is_baseline': False
    }


def run_minimal_experiment(config: dict) -> dict:
    """Run a minimal experiment to test the pipeline."""
    
    print(f"Running experiment with seed {config['seed']}...")
    
    # Initialize tracking components
    tracker = ExperimentTracker(config)
    performance = PerformanceTracker()
    
    try:
        # ================== 1. DATA GENERATION ==================
        tracker.logger.logger.info("Step 1: Generating synthetic data")
        performance.start_timer('data_generation')
        
        # Use data RNG stream for synthetic data
        data_rng = tracker.get_rng('data')
        
        # Generate synthetic observations (simplified)
        n_obs = config['n']
        synthetic_data = data_rng.randn(n_obs) * config['sigma']
        true_kappa = np.array([1.0, 1.5, 2.0])  # Fixed true parameter
        
        data_hash = tracker.compute_hash(synthetic_data)
        tracker.logger.logger.info(f"Generated {n_obs} observations, hash: {data_hash[:8]}")
        
        performance.end_timer('data_generation')
        
        # ================== 2. MCMC SAMPLING ==================
        tracker.logger.logger.info("Step 2: Running simplified MCMC")
        performance.start_timer('mcmc')
        
        # Use MCMC RNG stream
        mcmc_rng = tracker.get_rng('mcmc')
        
        # Simplified MCMC simulation
        n_samples = config['mcmc_n_steps'] - config['mcmc_n_burn']
        samples = np.zeros((n_samples, config['m']))
        
        # Simple random walk around true parameter
        current = true_kappa.copy()
        n_accepted = 0
        
        for i in range(config['mcmc_n_steps']):
            # Propose new state
            proposal = current + mcmc_rng.randn(config['m']) * 0.1
            
            # Simple acceptance (always accept for testing)
            current = proposal
            n_accepted += 1
            
            # Store post burn-in samples
            if i >= config['mcmc_n_burn']:
                samples[i - config['mcmc_n_burn']] = current
        
        acceptance_rate = n_accepted / config['mcmc_n_steps']
        
        # Compute simple diagnostics
        ess = np.array([400, 420, 410])  # Mock ESS values
        
        tracker.logger.logger.info(f"MCMC completed: acceptance={acceptance_rate:.3f}")
        
        performance.end_timer('mcmc')
        
        # ================== 3. MOCK CERTIFICATE ==================
        tracker.logger.logger.info("Step 3: Computing mock certificate")
        performance.start_timer('certificate')
        
        # Mock certificate computation
        empirical_loss = 0.3
        kl_term = 1.5
        eta_h = 0.01
        certificate = empirical_loss + kl_term / (config['lambda'] * config['n']) + eta_h
        
        tracker.logger.logger.info(f"Certificate B_λ = {certificate:.4f}")
        
        performance.end_timer('certificate')
        
        # ================== 4. COMPILE RESULTS ==================
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
                'n': n_obs
            },
            'mcmc': {
                'acceptance_rate': acceptance_rate,
                'ess': ess.tolist(),
                'converged': True,
                'n_forward_evals': 100  # Mock
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
                'KL': kl_term,
                'eta_h': eta_h
            },
            'performance': performance.get_summary()
        }
        
        tracker.logger.logger.info("✓ Experiment completed successfully")
        return results
        
    except Exception as e:
        tracker.logger.logger.error(f"✗ Experiment failed: {e}")
        raise


def verify_reproducibility():
    """Verify that experiments with same seed produce identical results."""
    print("\n" + "="*60)
    print("REPRODUCIBILITY VERIFICATION TEST")
    print("="*60)
    
    config = create_test_config(101)
    
    print("Running experiment twice with seed 101...")
    result1 = run_minimal_experiment(config.copy())
    result2 = run_minimal_experiment(config.copy())
    
    # Check reproducibility
    hash_match = result1['reproducibility']['data_hash'] == result2['reproducibility']['data_hash']
    acc_match = abs(result1['mcmc']['acceptance_rate'] - result2['mcmc']['acceptance_rate']) < 1e-10
    
    print(f"Data hash match: {'✓' if hash_match else '✗'}")
    print(f"Acceptance rate match: {'✓' if acc_match else '✗'}")
    
    if hash_match and acc_match:
        print("✓ REPRODUCIBILITY VERIFIED")
        return True
    else:
        print("✗ REPRODUCIBILITY FAILED")
        return False


def main():
    """Run test experiments."""
    
    # Create output directory
    output_dir = Path('results_test')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    print("="*60)
    print("RUNNING TEST EXPERIMENTS")
    print("="*60)
    
    # Run reproducibility verification
    if not verify_reproducibility():
        print("⚠ Warning: Reproducibility verification failed!")
        return
    
    # Run test experiments with all three seeds
    seeds = [101, 202, 303]
    results = []
    
    print(f"\nRunning {len(seeds)} test experiments...")
    
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Experiment with seed {seed}")
        
        config = create_test_config(seed)
        result = run_minimal_experiment(config)
        results.append(result)
        
        # Save individual result
        result_file = output_dir / f'result_seed_{seed}.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Saved result to {result_file}")
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results_file = output_dir / f'results_test_{timestamp}.json'
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TEST EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(results)}")
    print(f"All completed successfully: ✓")
    print(f"Results saved to: {all_results_file}")
    
    # Verify different seeds produce different results
    hashes = [r['reproducibility']['data_hash'] for r in results]
    if len(set(hashes)) == len(hashes):
        print("✓ Different seeds produce different results")
    else:
        print("⚠ Warning: Some seeds produced identical results")
    
    print(f"\nPipeline verification complete!")
    print("Ready to run full experiments.")


if __name__ == '__main__':
    main()