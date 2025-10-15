# run_classical_baseline.py
"""
Run classical Bayesian baseline experiments for comparison with PAC-Bayes results
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import numpy as np
from datetime import datetime
from src.inference.classical_posterior import ClassicalPosterior
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2
from src.forward_model.heat_equation import HeatEquationSolver
from src.inference.prior import Prior
from src.data.data_generator import DataGenerator
from src.utils.reproducibility import set_seed

def run_classical_experiment(config):
    """Run single classical baseline experiment"""
    set_seed(config['seed'])
    
    # Initialize components (same as PAC-Bayes experiments)
    solver = HeatEquationSolver(n_x=config['n_x'], T=config['T'])
    prior = Prior(m=config['m'], sigma_kappa=0.5)
    
    # Generate data
    generator = DataGenerator(solver=solver, sigma=config['sigma'])
    sensor_positions = generator.get_sensor_positions(config['s'], placement_type='fixed')
    time_grid = np.linspace(0, config['T'], 150)
    
    # Generate ground truth and observations
    kappa_true = prior.sample()
    y = generator.generate_observations(kappa_true, sensor_positions, time_grid)
    
    # Create classical posterior (NO temperature parameter)
    posterior = ClassicalPosterior(
        y=y, 
        solver=solver, 
        sigma=config['sigma'],
        prior=prior,
        sensor_positions=sensor_positions,
        time_grid=time_grid,
        n=len(y)
    )
    
    # Run MCMC (same setup as PAC-Bayes)
    sampler = AdaptiveMetropolisHastingsPhase2(
        target_log_prob=posterior.log_posterior,
        initial_cov=np.eye(config['m']) * 0.1,
        target_accept_rate=0.234
    )
    
    # Sample from classical posterior
    n_samples = 10000
    n_burn = 5000
    
    samples = sampler.sample(
        n_samples=n_samples + n_burn,
        initial_state=prior.sample()
    )
    
    # Keep post-burn-in samples
    posterior_samples = samples[n_burn:]
    
    return {
        'config': config,
        'samples': posterior_samples.tolist(),
        'acceptance_rate': sampler.acceptance_rate,
        'status': 'success'
    }

def main():
    print("=" * 60)
    print("CLASSICAL BAYESIAN BASELINE EXPERIMENTS")
    print("=" * 60)
    
    # Configure for baseline subset (reduced for comparison)
    baseline_configs = []
    for s in [3, 5]:
        for sigma in [0.05, 0.10, 0.20]:
            for m in [3, 5]:
                for seed in [101]:  # Single seed for baseline
                    config = {
                        's': s,
                        'sigma': sigma,
                        'n_x': 100,  # Fixed at 100
                        'T': 0.5,    # Fixed at 0.5
                        'm': m,
                        'seed': seed
                    }
                    baseline_configs.append(config)
    
    print(f"Running {len(baseline_configs)} classical baseline experiments")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, config in enumerate(baseline_configs):
        print(f"\nExperiment {i+1}/{len(baseline_configs)}: s={config['s']}, σ={config['sigma']}, m={config['m']}")
        
        try:
            result = run_classical_experiment(config)
            results.append(result)
            print(f"  ✅ Success (acceptance rate: {result['acceptance_rate']:.3f})")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'config': config,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save results
    output_file = f'classical_baseline_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    success_rate = sum(1 for r in results if r['status'] == 'success') / len(results)
    print(f"\n" + "=" * 60)
    print(f"CLASSICAL BASELINE COMPLETE")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()