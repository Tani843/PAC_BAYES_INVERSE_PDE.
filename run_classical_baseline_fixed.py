# run_classical_baseline_fixed.py
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

def run_classical_baseline_fixed():
    """
    Section K: Classical baseline with correct MCMC API
    """
    
    output_dir = Path(f'classical_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(exist_ok=True)
    
    baseline_configs = []
    config_gen = ExperimentConfig()
    
    for s in [3, 5]:
        placement = 'fixed'
        sensor_positions = config_gen.sensor_placements[s][placement]
        
        for sigma in [0.05, 0.10, 0.20]:
            for m in [3, 5]:
                for seed in [101, 202, 303]:
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
                        'delta': 0.05,
                        'alpha': 1e-3,
                        'M': 2000,
                        'R': 100,
                        'c': 1.0,
                        'n': s * 50
                    }
                    baseline_configs.append(config)
    
    print(f"Running {len(baseline_configs)} classical baseline experiments")
    results = []
    
    for i, config in enumerate(baseline_configs):
        print(f"\n[{i+1}/{len(baseline_configs)}] Classical baseline")
        print(f"  s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}, seed={config['seed']}")
        
        try:
            # Generate data
            data_gen = DataGenerator(config)
            dataset = data_gen.generate_dataset()
            
            # Setup solver
            solver = HeatEquationSolver(
                n_x=config['n_x'],
                n_t=config['n_t'],
                T=config['T']
            )
            
            # Setup prior
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
            
            # FIX: Pass adaptive parameters to constructor
            sampler = AdaptiveMetropolisHastingsPhase2(
                posterior=posterior,
                seed=config['seed'],
                ess_target=200,      # Pass to constructor
                chunk_size=2000,     # Pass to constructor
                max_steps=10000,     # Pass to constructor
                use_block_updates=True  # Optional: enable block updates
            )
            
            # FIX: Call run_adaptive_length with only n_burn
            mcmc_results = sampler.run_adaptive_length(n_burn=1000)
            
            # Compute bounded loss for comparison
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
                'mcmc': {
                    'acceptance_rate': mcmc_results.get('overall_acceptance_rate', 0),  # FIXED: correct key
                    'min_ess': float(np.min(mcmc_results.get('final_ess', [0]))),  # FIXED: correct key and extraction
                    'converged': mcmc_results.get('converged', False),
                    'n_samples': len(samples)
                },
                'credible_intervals': credible_intervals,
                'bounded_loss_mean': np.mean(bounded_losses)
            }
            
            print(f"  ✓ Success: acc={result['mcmc']['acceptance_rate']:.1%}")
            
        except Exception as e:
            result = {'config': config, 'status': 'error', 'error': str(e)}
            print(f"  ✗ Error: {e}")
        
        results.append(result)
        
        # Save periodically
        if (i + 1) % 10 == 0:
            with open(output_dir / f'checkpoint_{i+1}.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final
    with open(output_dir / 'classical_baseline_72configs.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Completed {len(baseline_configs)} experiments")
    return results

if __name__ == '__main__':
    run_classical_baseline_fixed()