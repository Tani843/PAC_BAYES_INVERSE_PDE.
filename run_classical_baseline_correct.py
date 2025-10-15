# run_classical_baseline_correct.py
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from config.experiment_config import ExperimentConfig
from src.forward_model.heat_equation import HeatEquationSolver
from src.data.data_generator import DataGenerator
from src.inference.loss_functions import Prior, BoundedLoss
from src.inference.classical_posterior import ClassicalPosterior
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

def run_classical_baseline_correct():
    """
    Section K: Classical baseline with EXACT same setup as PAC-Bayes
    72 configs: s∈{3,5}, σ∈{0.05,0.10,0.20}, m∈{3,5}, seeds∈{101,202,303}
    Fixed: n_x=100, T=0.5, placement='fixed'
    NO λ (classical doesn't use temperature parameter)
    """
    
    output_dir = Path(f'classical_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(exist_ok=True)
    
    baseline_configs = []
    config_gen = ExperimentConfig()
    
    # 72 configurations total (no λ variation for classical)
    for s in [3, 5]:
        placement = 'fixed'  # Section K specifies fixed sensors
        sensor_positions = config_gen.sensor_placements[s][placement]
        
        for sigma in [0.05, 0.10, 0.20]:
            for m in [3, 5]:
                for seed in [101, 202, 303]:
                    config = {
                        's': s,
                        'sensor_positions': sensor_positions,
                        'placement_type': placement,
                        'sigma': sigma,
                        'n_x': 100,  # Section K: n_x=100
                        'n_t': 50,   
                        'T': 0.5,    # Section K: T=0.5
                        'm': m,
                        'seed': seed,
                        'delta': 0.05,
                        'alpha': 1e-3,
                        'M': 2000,   # For certificate comparison
                        'R': 100,    # For L_MC
                        'c': 1.0,    # For bounded loss evaluation
                        'n': s * 50  # Total observations
                    }
                    baseline_configs.append(config)
    
    print(f"Running {len(baseline_configs)} classical baseline experiments")
    print("Configuration: n_x=100, T=0.5, fixed sensors only")
    
    results = []
    
    for i, config in enumerate(baseline_configs):
        print(f"\n[{i+1}/{len(baseline_configs)}] Classical baseline")
        print(f"  s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}, seed={config['seed']}")
        
        try:
            # Generate data (SAME as PAC-Bayes)
            data_gen = DataGenerator(config)
            dataset = data_gen.generate_dataset()
            
            # Setup solver (SAME as PAC-Bayes)
            solver = HeatEquationSolver(
                n_x=config['n_x'],
                n_t=config['n_t'],
                T=config['T']
            )
            
            # Setup prior (SAME as PAC-Bayes)
            prior = Prior(
                m=config['m'],
                mu_0=0.0,
                tau2=1.0,
                kappa_min=0.1,
                kappa_max=5.0
            )
            
            # Classical posterior
            posterior = ClassicalPosterior(
                y=dataset['noisy_data'],
                solver=solver,
                sigma=config['sigma'],
                prior=prior,
                sensor_positions=dataset['sensor_positions'],
                time_grid=dataset['time_grid'],
                n=config['n']
            )
            
            # Run MCMC (SAME sampler as PAC-Bayes)
            sampler = AdaptiveMetropolisHastingsPhase2(
                posterior=posterior,
                seed=config['seed']
            )
            
            mcmc_results = sampler.run_adaptive_length(
                target_ess=200,
                chunk_size=2000,
                max_steps=10000,
                n_burn=1000
            )
            
            # Compute bounded loss on classical samples for comparison
            loss_fn = BoundedLoss(c=config['c'], sigma=config['sigma'])
            bounded_losses = []
            for sample in mcmc_results['samples']:
                F_sample = posterior.forward_map(sample)
                loss_val = loss_fn.compute_empirical_loss(dataset['noisy_data'], F_sample)
                bounded_losses.append(loss_val)
            
            # Extract credible intervals
            samples = mcmc_results['samples']
            credible_intervals = {}
            for j in range(config['m']):
                param_samples = samples[:, j]
                credible_intervals[f'kappa_{j}'] = {
                    'mean': np.mean(param_samples),
                    'q025': np.percentile(param_samples, 2.5),
                    'q975': np.percentile(param_samples, 97.5)
                }
            
            result = {
                'config': config,
                'status': 'success',
                'mcmc': mcmc_results,
                'credible_intervals': credible_intervals,
                'bounded_loss_mean': np.mean(bounded_losses),  # For comparison
                'bounded_loss_std': np.std(bounded_losses)
            }
            
            print(f"  ✓ Success: acc={mcmc_results['acceptance_rate']:.1%}, "
                  f"ESS={mcmc_results['min_ess']:.0f}")
            
        except Exception as e:
            result = {'config': config, 'status': 'error', 'error': str(e)}
            print(f"  ✗ Error: {e}")
        
        results.append(result)
    
    # Save results
    with open(output_dir / 'classical_baseline_72configs.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Completed {len(results)} classical baseline experiments")
    return results

if __name__ == '__main__':
    run_classical_baseline_correct()