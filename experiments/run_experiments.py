"""
Main experiment runner for PAC-Bayes certification experiments
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import ExperimentConfig
from src.forward_model.heat_equation import HeatEquationSolver
from src.data.data_generator import DataGenerator
from src.inference.loss_functions import BoundedLoss, Prior
from src.inference.gibbs_posterior import GibbsPosterior, ClassicalPosterior
from src.mcmc.metropolis_hastings import MetropolisHastings
from src.certificate.pac_bayes_bound import PACBayesCertificate

def run_single_experiment(config_dict: Dict) -> Dict:
    """
    Run a single experiment configuration.
    
    Args:
        config_dict: Experiment configuration dictionary
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: s={config_dict['s']}, Ã={config_dict['sigma']}, "
          f"»={config_dict.get('lambda', 'Bayes')}, seed={config_dict['seed']}")
    print(f"{'='*60}")
    
    # 1. Generate data
    print("Generating data...")
    data_gen = DataGenerator(config_dict)
    dataset = data_gen.generate_dataset()
    
    # 2. Setup solver and loss
    solver = HeatEquationSolver(
        n_x=config_dict['n_x'],
        n_t=config_dict['n_t'],
        T=config_dict['T']
    )
    
    loss_fn = BoundedLoss(
        c=config_dict['c'],
        sigma=config_dict['sigma']
    )
    
    prior = Prior(
        m=config_dict['m'],
        mu_0=0.0,
        tau2=1.0,
        kappa_min=0.1,
        kappa_max=5.0
    )
    
    # 3. Setup posterior
    if config_dict.get('is_baseline', False):
        # Classical Bayesian posterior
        print("Setting up Classical Bayesian posterior...")
        posterior = ClassicalPosterior(
            y=dataset['noisy_data'],
            solver=solver,
            prior=prior,
            sigma=config_dict['sigma'],
            sensor_positions=dataset['sensor_positions'],
            time_grid=dataset['time_grid']
        )
    else:
        # Gibbs posterior with temperature »
        print(f"Setting up Gibbs posterior with »={config_dict['lambda']}...")
        posterior = GibbsPosterior(
            y=dataset['noisy_data'],
            solver=solver,
            loss_fn=loss_fn,
            prior=prior,
            lambda_val=config_dict['lambda'],
            n=config_dict['n'],
            sensor_positions=dataset['sensor_positions'],
            time_grid=dataset['time_grid']
        )
    
    # 4. Run MCMC
    print("Running MCMC...")
    sampler = MetropolisHastings(
        posterior=posterior,
        proposal_scale=0.1,
        seed=config_dict['seed']
    )
    
    mcmc_results = sampler.run_chain(
        n_steps=10000,
        n_burn=2000,
        thin=1,
        verbose=False
    )
    
    # Check convergence
    if not mcmc_results['converged']['converged']:
        print(f"Warning: MCMC did not converge! Min ESS: {np.min(mcmc_results['ess']):.1f}")
    
    # 5. Compute certificate (if not baseline)
    if not config_dict.get('is_baseline', False):
        print("Computing PAC-Bayes certificate...")
        certificate = PACBayesCertificate(
            delta=config_dict['delta'],
            alpha=config_dict['alpha'],
            M=config_dict['M'],
            R=config_dict['R']
        )
        
        # Empirical term
        empirical_dict = certificate.compute_empirical_term(
            mcmc_results['samples'],
            mcmc_results['losses']
        )
        
        # KL term (need prior samples)
        print("Estimating partition function...")
        prior_samples = prior.sample(config_dict['M'], seed=config_dict['seed'] + 5000)
        prior_losses = np.zeros(config_dict['M'])
        
        for i, kappa in enumerate(prior_samples):
            F_kappa = posterior.forward_map(kappa)
            prior_losses[i] = loss_fn.compute_empirical_loss(dataset['noisy_data'], F_kappa)
        
        kl_dict = certificate.compute_kl_term(
            mcmc_results['losses'],
            prior_samples,
            prior_losses,
            config_dict['lambda'],
            config_dict['n']
        )
        
        # Discretization penalty
        print("Computing discretization penalty...")
        # Sample a few posterior samples for testing
        test_samples = mcmc_results['samples'][:10]
        
        # Create refined solver if needed
        solver_h2 = HeatEquationSolver(
            n_x=config_dict['n_x'] * 2,
            n_t=config_dict['n_t'] * 2,
            T=config_dict['T']
        )
        
        discretization_dict = certificate.compute_discretization_penalty(
            test_samples,
            solver,
            solver_h2,
            loss_fn,
            dataset['noisy_data'],
            dataset['sensor_positions'],
            dataset['time_grid']
        )
        
        # Final bound
        bound_dict = certificate.compute_final_bound(
            empirical_dict,
            kl_dict,
            discretization_dict,
            config_dict['lambda'],
            config_dict['n']
        )
        
        # True risk with fresh noise (optional, expensive)
        print("Computing true risk with fresh noise...")
        fresh_replicates = data_gen.generate_fresh_noise_replicates(
            dataset['noiseless_data'],
            R=config_dict['R']
        )
        
        true_risk_dict = certificate.compute_true_risk_mc(
            mcmc_results['samples'][:100],  # Use subset for speed
            fresh_replicates,
            solver,
            loss_fn,
            dataset['sensor_positions'],
            dataset['time_grid']
        )
    else:
        # For baseline, just compute credible intervals
        bound_dict = None
        true_risk_dict = None
    
    # 6. Compile results
    results = {
        'config': config_dict,
        'dataset': {
            'kappa_star': dataset['kappa_star'].tolist(),
            'sigma': dataset['sigma'],
            'n': dataset['n']
        },
        'mcmc': {
            'acceptance_rate': mcmc_results['acceptance_rate'],
            'ess': mcmc_results['ess'].tolist(),
            'converged': mcmc_results['converged'],
            'n_forward_evals': mcmc_results['n_forward_evals']
        },
        'posterior_summary': {
            'mean': np.mean(mcmc_results['samples'], axis=0).tolist(),
            'std': np.std(mcmc_results['samples'], axis=0).tolist(),
            'quantiles': {
                '2.5%': np.percentile(mcmc_results['samples'], 2.5, axis=0).tolist(),
                '50%': np.percentile(mcmc_results['samples'], 50, axis=0).tolist(),
                '97.5%': np.percentile(mcmc_results['samples'], 97.5, axis=0).tolist()
            }
        }
    }
    
    if bound_dict is not None:
        results['certificate'] = {
            'B_lambda': bound_dict['B_lambda'],
            'L_hat': bound_dict['L_hat'],
            'KL': bound_dict['KL'],
            'eta_h': bound_dict['eta_h'],
            'components': bound_dict['components']
        }
    
    if true_risk_dict is not None:
        results['true_risk'] = {
            'L_mc': true_risk_dict['L_mc'],
            'L_mc_std': true_risk_dict['L_mc_std']
        }
    
    print(f"Experiment complete!")
    if bound_dict is not None:
        print(f"Certificate B_» = {bound_dict['B_lambda']:.4f}")
        print(f"  L = {bound_dict['L_hat']:.4f}")
        print(f"  KL/(»n) = {bound_dict['components']['kl_term']:.4f}")
        print(f"  ·_h = {bound_dict['eta_h']:.4f}")
    
    return results


def run_main_experiments(config: ExperimentConfig, 
                        subset: str = 'main',
                        output_dir: str = 'results'):
    """
    Run the main experiment grid.
    
    Args:
        config: Experiment configuration object
        subset: 'main', 'baseline', or 'test'
        output_dir: Directory for saving results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'logs').mkdir(exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)
    (output_path / 'tables').mkdir(exist_ok=True)
    (output_path / 'checkpoints').mkdir(exist_ok=True)
    
    # Get experiment list
    if subset == 'main':
        experiments = config.get_experiment_grid(include_appendix=False)
    elif subset == 'baseline':
        experiments = config.get_baseline_subset()
    elif subset == 'test':
        # Small test subset
        experiments = config.get_experiment_grid()[:3]
    else:
        raise ValueError(f"Unknown subset: {subset}")
    
    print(f"Running {len(experiments)} experiments...")
    
    # Run experiments
    all_results = []
    for i, exp_config in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting experiment...")
        
        try:
            result = run_single_experiment(exp_config)
            all_results.append(result)
            
            # Save intermediate results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = output_path / 'checkpoints' / f'result_{i}_{timestamp}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error in experiment {i}: {e}")
            continue
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON format
    results_json = output_path / f'results_{subset}_{timestamp}.json'
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Pickle format (preserves numpy arrays)
    results_pkl = output_path / f'results_{subset}_{timestamp}.pkl'
    with open(results_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nAll experiments complete!")
    print(f"Results saved to {results_json}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run PAC-Bayes experiments')
    parser.add_argument('--subset', type=str, default='test',
                       choices=['main', 'baseline', 'test'],
                       help='Which experiment subset to run')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig()
    
    # Validate configuration
    if not config.validate_config():
        raise ValueError("Invalid configuration!")
    
    # Log configuration
    print(config.log_config())
    
    # Run experiments
    results = run_main_experiments(config, subset=args.subset, output_dir=args.output)
    
    print(f"\nCompleted {len(results)} experiments successfully!")


if __name__ == '__main__':
    main()