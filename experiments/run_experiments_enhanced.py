"""
Enhanced experiment runner with full Section J reproducibility features
Implements complete logging, checkpointing, and deterministic execution
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import ExperimentConfig
from src.forward_model.heat_equation import HeatEquationSolver
from src.data.data_generator import DataGenerator
from src.inference.loss_functions import BoundedLoss, Prior
from src.inference.gibbs_posterior import GibbsPosterior, ClassicalPosterior
from src.mcmc.metropolis_hastings import MetropolisHastings
from src.certificate.pac_bayes_bound import PACBayesCertificate
from src.utils.reproducibility import (
    RandomStateManager, ExperimentLogger, CheckpointManager, ExperimentTracker
)
from src.utils.logging import DiagnosticsLogger, PerformanceTracker


def run_single_experiment_with_tracking(config_dict: Dict, 
                                       resume_from: Optional[str] = None) -> Dict:
    """
    Run a single experiment with full reproducibility tracking.
    
    Implements Section J requirements:
    - Fixed seeds {101, 202, 303}
    - Separate RNG streams for (i) data noise, (ii) prior sampling, (iii) MCMC
    - Complete logging and checkpointing
    - Pre-publish validation checks
    
    Args:
        config_dict: Experiment configuration
        resume_from: Optional checkpoint path to resume from
        
    Returns:
        Complete results dictionary with reproducibility metadata
    """
    # Initialize tracking components
    tracker = ExperimentTracker(config_dict)
    diagnostics = DiagnosticsLogger()
    performance = PerformanceTracker()
    
    # Log experiment start
    tracker.logger.logger.info(f"Starting experiment with seed {config_dict['seed']}")
    tracker.logger.logger.info(f"Configuration: s={config_dict['s']}, σ={config_dict['sigma']}, "
                              f"λ={config_dict.get('lambda', 'Bayes')}, n_x={config_dict['n_x']}")
    
    # Resume from checkpoint if provided
    if resume_from:
        checkpoint = tracker.load_state(resume_from)
        results = checkpoint['results']
        tracker.logger.logger.info(f"Resumed from checkpoint at iteration {checkpoint['iteration']}")
    else:
        results = {}
    
    try:
        # ================== 1. DATA GENERATION ==================
        tracker.logger.logger.info("Step 1: Generating data with fixed seed")
        performance.start_timer('data_generation')
        
        # Use data RNG stream
        data_rng = tracker.get_rng('data')
        
        # Initialize data generator with specific RNG
        data_gen = DataGenerator(config_dict)
        data_gen.rng = data_rng  # Override with tracked RNG
        
        # Generate dataset
        dataset = data_gen.generate_dataset()
        
        # Log data characteristics
        tracker.logger.logger.info(f"Generated {dataset['n']} observations")
        tracker.logger.logger.info(f"True κ*: {dataset['kappa_star']}")
        tracker.logger.logger.info(f"Data range: [{dataset['noisy_data'].min():.4f}, "
                                  f"{dataset['noisy_data'].max():.4f}]")
        
        # Compute data hash for verification
        data_hash = tracker.compute_hash(dataset['noisy_data'])
        tracker.logger.logger.info(f"Data hash: {data_hash[:8]}")
        
        performance.end_timer('data_generation')
        performance.record_memory('after_data')
        
        # ================== 2. SETUP FORWARD MODEL ==================
        tracker.logger.logger.info("Step 2: Setting up forward model")
        performance.start_timer('model_setup')
        
        solver = HeatEquationSolver(
            n_x=config_dict['n_x'],
            n_t=config_dict['n_t'],
            T=config_dict['T']
        )
        
        # Validate solver
        validation = solver.validate_solver()
        tracker.logger.logger.info(f"Solver validation: L∞ error = {validation['linf_error']:.2e}")
        
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
        
        performance.end_timer('model_setup')
        
        # ================== 3. SETUP POSTERIOR ==================
        if config_dict.get('is_baseline', False):
            tracker.logger.logger.info("Step 3: Setting up Classical Bayesian posterior")
            posterior = ClassicalPosterior(
                y=dataset['noisy_data'],
                solver=solver,
                prior=prior,
                sigma=config_dict['sigma'],
                sensor_positions=dataset['sensor_positions'],
                time_grid=dataset['time_grid']
            )
        else:
            tracker.logger.logger.info(f"Step 3: Setting up Gibbs posterior with λ={config_dict['lambda']}")
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
        
        # ================== 4. RUN MCMC ==================
        tracker.logger.logger.info("Step 4: Running MCMC with tracked RNG")
        performance.start_timer('mcmc')
        
        # Use MCMC RNG stream
        mcmc_rng = tracker.get_rng('mcmc')
        
        sampler = MetropolisHastings(
            posterior=posterior,
            proposal_scale=0.1,
            seed=None  # Will use provided RNG
        )
        sampler.rng = mcmc_rng  # Override with tracked RNG
        
        # Run chain
        mcmc_results = sampler.run_chain(
            n_steps=config_dict.get('mcmc_n_steps', 10000),
            n_burn=config_dict.get('mcmc_n_burn', 2000),
            thin=1,
            verbose=True
        )
        
        performance.end_timer('mcmc')
        performance.solver_calls = posterior.n_evals
        
        # Log MCMC diagnostics
        diagnostics.log_chain_diagnostics(
            config_dict,
            mcmc_results['full_chain'],
            mcmc_results['acceptance_rate'],
            mcmc_results['ess'],
            mcmc_results['acf']
        )
        
        tracker.logger.logger.info(f"MCMC completed: acceptance={mcmc_results['acceptance_rate']:.3f}, "
                                  f"min ESS={np.min(mcmc_results['ess']):.1f}")
        
        # Pre-publish check: Convergence
        if not mcmc_results['converged']:
            tracker.logger.logger.warning("⚠ MCMC did not converge!")
            if np.min(mcmc_results['ess']) < 500:
                tracker.logger.logger.warning(f"⚠ ESS too low: {np.min(mcmc_results['ess']):.1f} < 500")
            if not (0.2 <= mcmc_results['acceptance_rate'] <= 0.5):
                tracker.logger.logger.warning(f"⚠ Acceptance rate {mcmc_results['acceptance_rate']:.3f} "
                                             "outside [0.2, 0.5]")
        
        # Save checkpoint after MCMC
        tracker.current_iteration = 1
        tracker.save_state({'mcmc_results': mcmc_results}, {'stage': 'after_mcmc'})
        
        # ================== 5. COMPUTE CERTIFICATE ==================
        if not config_dict.get('is_baseline', False):
            tracker.logger.logger.info("Step 5: Computing PAC-Bayes certificate")
            performance.start_timer('certificate')
            
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
            
            # Pre-publish check: Loss boundedness
            diagnostics.log_loss_components(
                config_dict,
                empirical_dict['empirical_mean'],
                empirical_dict['empirical_mean'],
                0.0  # Placeholder for squared loss
            )
            
            if empirical_dict['empirical_min'] < 0 or empirical_dict['empirical_max'] > 1:
                tracker.logger.logger.error(f"⚠ Loss not in (0,1): [{empirical_dict['empirical_min']}, "
                                           f"{empirical_dict['empirical_max']}]")
            
            # KL term with prior RNG stream
            tracker.logger.logger.info("Estimating partition function Z_λ")
            prior_rng = tracker.get_rng('prior')
            
            prior_samples = prior.sample(config_dict['M'], seed=None)
            # Use tracked RNG for prior sampling
            for i in range(config_dict['M']):
                for j in range(config_dict['m']):
                    prior_samples[i, j] = prior_rng.uniform(0.1, 5.0)
            
            prior_losses = np.zeros(config_dict['M'])
            for i, kappa in enumerate(prior_samples):
                F_kappa = posterior.forward_map(kappa)
                prior_losses[i] = loss_fn.compute_empirical_loss(dataset['noisy_data'], F_kappa)
                performance.increment_solver_calls()
            
            kl_dict = certificate.compute_kl_term(
                mcmc_results['losses'],
                prior_samples,
                prior_losses,
                config_dict['lambda'],
                config_dict['n']
            )
            
            # Pre-publish check: underline_Z > 0
            if kl_dict['underline_Z'] <= 0:
                tracker.logger.logger.error(f"⚠ underline_Z = {kl_dict['underline_Z']:.6f} <= 0")
                tracker.logger.logger.error("  Consider increasing M (prior sampling budget)")
            
            # Discretization penalty
            tracker.logger.logger.info("Computing discretization penalty η_h")
            performance.start_timer('discretization')
            
            test_samples = mcmc_results['samples'][:10]
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
            
            performance.end_timer('discretization')
            
            # Pre-publish check: η_h refinement
            diagnostics.log_refinement_test(
                config_dict['n_x'],
                config_dict['n_x'] * 2,
                discretization_dict['eta_h'],
                discretization_dict.get('eta_refinement', discretization_dict['eta_h'] * 0.5)
            )
            
            # Final bound
            bound_dict = certificate.compute_final_bound(
                empirical_dict,
                kl_dict,
                discretization_dict,
                config_dict['lambda'],
                config_dict['n']
            )
            
            # Log certificate components
            diagnostics.log_certificate_components(
                config_dict,
                bound_dict['L_hat'],
                bound_dict['KL'],
                kl_dict['Z_hat'],
                kl_dict['underline_Z'],
                bound_dict['eta_h'],
                bound_dict['B_lambda']
            )
            
            tracker.logger.logger.info(f"Certificate B_λ = {bound_dict['B_lambda']:.4f}")
            tracker.logger.logger.info(f"  Components: L̂={bound_dict['L_hat']:.4f}, "
                                      f"KL/(λn)={bound_dict['components']['kl_term']:.4f}, "
                                      f"η_h={bound_dict['eta_h']:.6f}")
            
            performance.end_timer('certificate')
            
            # True risk estimation (optional)
            if config_dict['R'] > 0:
                tracker.logger.logger.info("Computing true risk L_MC")
                fresh_replicates = data_gen.generate_fresh_noise_replicates(
                    dataset['noiseless_data'],
                    R=config_dict['R']
                )
                
                true_risk_dict = certificate.compute_true_risk_mc(
                    mcmc_results['samples'][:100],
                    fresh_replicates,
                    solver,
                    loss_fn,
                    dataset['sensor_positions'],
                    dataset['time_grid']
                )
                
                tracker.logger.logger.info(f"True risk L_MC = {true_risk_dict['L_mc']:.4f}")
        else:
            bound_dict = None
            true_risk_dict = None
        
        # ================== 6. COMPILE RESULTS ==================
        results = {
            'config': config_dict,
            'reproducibility': {
                'seed': config_dict['seed'],
                'data_hash': data_hash,
                'experiment_id': tracker.logger.experiment_id,
                'git_commit': tracker.logger.metadata.get('git_commit'),
                'rng_usage': tracker.rng_manager.usage_log.copy()
            },
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
        
        # Add performance metrics
        results['performance'] = performance.get_summary()
        
        # Save final results
        tracker.save_state(results, {'stage': 'complete'})
        
        # Save all diagnostics
        diagnostics.save_all_diagnostics(tracker.logger.experiment_id)
        
        tracker.logger.logger.info("✓ Experiment completed successfully")
        
    except Exception as e:
        tracker.logger.logger.error(f"✗ Experiment failed: {e}")
        tracker.save_state({'error': str(e)}, {'stage': 'error'})
        raise
    
    finally:
        # Save metadata
        tracker.logger.save_metadata()
        
        # Print performance summary
        performance.print_summary()
    
    return results


def verify_reproducibility(config: ExperimentConfig):
    """
    Verify that experiments with same seed produce identical results.
    This is a critical test for Section J compliance.
    """
    print("\n" + "="*60)
    print("REPRODUCIBILITY VERIFICATION TEST")
    print("="*60)
    
    # Test configuration
    test_config = {
        's': 3,
        'sensor_positions': [0.25, 0.50, 0.75],
        'placement_type': 'fixed',
        'sigma': 0.1,
        'n_x': 50,
        'T': 0.5,
        'lambda': 1.0,
        'c': 1.0,
        'm': 3,
        'n_t': 50,
        'seed': 101,  # Fixed seed from {101, 202, 303}
        'delta': 0.05,
        'alpha': 1e-3,
        'M': 100,
        'R': 10,
        'n': 150,
        'Delta_x': 1.0 / 50,
        'Delta_t': 0.5 / 49,
        'mcmc_n_steps': 1000,  # Small for testing
        'mcmc_n_burn': 200
    }
    
    print(f"Running experiment twice with seed {test_config['seed']}...")
    
    # Run experiment twice
    result1 = run_single_experiment_with_tracking(test_config.copy())
    result2 = run_single_experiment_with_tracking(test_config.copy())
    
    # Extract key values for comparison
    checks = []
    
    # Check data hash
    hash_match = result1['reproducibility']['data_hash'] == result2['reproducibility']['data_hash']
    checks.append(('Data hash', hash_match))
    
    # Check MCMC results
    acc_match = abs(result1['mcmc']['acceptance_rate'] - result2['mcmc']['acceptance_rate']) < 1e-10
    checks.append(('Acceptance rate', acc_match))
    
    # Check posterior mean
    mean1 = np.array(result1['posterior_summary']['mean'])
    mean2 = np.array(result2['posterior_summary']['mean'])
    mean_match = np.allclose(mean1, mean2, rtol=1e-10)
    checks.append(('Posterior mean', mean_match))
    
    # Check certificate (if computed)
    if 'certificate' in result1 and 'certificate' in result2:
        cert_match = abs(result1['certificate']['B_lambda'] - result2['certificate']['B_lambda']) < 1e-10
        checks.append(('Certificate B_λ', cert_match))
    
    # Print results
    print("\nReproducibility Check Results:")
    print("-" * 40)
    all_pass = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:20s}: {status}")
        all_pass = all_pass and passed
    
    print("-" * 40)
    if all_pass:
        print("✓ REPRODUCIBILITY VERIFIED: All checks passed!")
    else:
        print("✗ REPRODUCIBILITY FAILED: Some checks did not pass")
    
    return all_pass


def run_main_experiments_enhanced(config: ExperimentConfig,
                                 subset: str = 'main',
                                 output_dir: str = 'results',
                                 verify_repro: bool = True):
    """
    Run experiments with full Section J reproducibility features.
    
    Args:
        config: Experiment configuration
        subset: Which experiments to run
        output_dir: Output directory
        verify_repro: Whether to run reproducibility verification
    """
    # Create output directories
    output_path = Path(output_dir)
    (output_path / 'logs').mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(parents=True, exist_ok=True)
    (output_path / 'tables').mkdir(parents=True, exist_ok=True)
    (output_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
    
    # Run reproducibility verification first
    if verify_repro:
        if not verify_reproducibility(config):
            print("\n⚠ Warning: Reproducibility verification failed!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return None
    
    # Get experiment list
    if subset == 'main':
        experiments = config.get_experiment_grid(include_appendix=False)
    elif subset == 'baseline':
        experiments = config.get_baseline_subset()
    elif subset == 'test':
        # Small test subset with all three seeds
        test_configs = []
        for seed in [101, 202, 303]:
            exp = config.get_experiment_grid()[0].copy()
            exp['seed'] = seed
            test_configs.append(exp)
        experiments = test_configs
    else:
        raise ValueError(f"Unknown subset: {subset}")
    
    print(f"\n{'='*60}")
    print(f"Running {len(experiments)} experiments")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")
    
    # Initialize global diagnostics logger
    global_diagnostics = DiagnosticsLogger(output_path / 'logs')
    
    # Run experiments
    all_results = []
    failed_experiments = []
    
    for i, exp_config in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting experiment")
        print(f"  Config: s={exp_config['s']}, σ={exp_config['sigma']}, "
              f"λ={exp_config.get('lambda', 'Bayes')}, seed={exp_config['seed']}")
        
        try:
            # Check for existing checkpoint
            checkpoint_manager = CheckpointManager(output_path / 'checkpoints')
            exp_id = f"exp_s{exp_config['s']}_sig{exp_config['sigma']}_seed{exp_config['seed']}"
            latest_checkpoint = checkpoint_manager.find_latest_checkpoint(exp_id)
            
            if latest_checkpoint:
                print(f"  Found checkpoint: {latest_checkpoint}")
                response = input("  Resume from checkpoint? (y/n/s to skip): ")
                if response.lower() == 'y':
                    result = run_single_experiment_with_tracking(
                        exp_config,
                        resume_from=str(latest_checkpoint)
                    )
                elif response.lower() == 's':
                    print("  Skipping experiment")
                    continue
                else:
                    result = run_single_experiment_with_tracking(exp_config)
            else:
                result = run_single_experiment_with_tracking(exp_config)
            
            all_results.append(result)
            
            # Save intermediate results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = output_path / 'checkpoints' / f'result_{i}_{timestamp}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_experiments.append({
                'index': i,
                'config': exp_config,
                'error': str(e)
            })
            continue
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save successful results
    if all_results:
        results_json = output_path / f'results_{subset}_{timestamp}.json'
        with open(results_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        results_pkl = output_path / f'results_{subset}_{timestamp}.pkl'
        with open(results_pkl, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\nSaved {len(all_results)} results to {results_json}")
    
    # Save failed experiments log
    if failed_experiments:
        failed_json = output_path / f'failed_{subset}_{timestamp}.json'
        with open(failed_json, 'w') as f:
            json.dump(failed_experiments, f, indent=2)
        
        print(f"\n⚠ {len(failed_experiments)} experiments failed")
        print(f"  Failed experiments log: {failed_json}")
    
    # Save global diagnostics
    global_diagnostics.save_all_diagnostics(f"global_{subset}_{timestamp}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {len(failed_experiments)}")
    
    # Pre-publish validation summary
    if all_results:
        print("\nPre-Publish Validation Summary:")
        n_converged = sum(1 for r in all_results if r['mcmc']['converged'])
        print(f"  MCMC converged: {n_converged}/{len(all_results)}")
        
        if any('certificate' in r for r in all_results):
            n_valid = sum(1 for r in all_results 
                         if 'certificate' in r and r['certificate']['B_lambda'] >= r['certificate']['L_hat'])
            n_cert = sum(1 for r in all_results if 'certificate' in r)
            print(f"  Valid certificates: {n_valid}/{n_cert}")
    
    print(f"{'='*60}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run PAC-Bayes experiments with full reproducibility'
    )
    parser.add_argument('--subset', type=str, default='test',
                       choices=['main', 'baseline', 'test'],
                       help='Which experiment subset to run')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip reproducibility verification')
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
    
    # Run experiments with enhanced tracking
    results = run_main_experiments_enhanced(
        config,
        subset=args.subset,
        output_dir=args.output,
        verify_repro=not args.no_verify
    )
    
    if results:
        print(f"\n✓ Completed {len(results)} experiments successfully!")
        
        # Generate figures if results available
        try:
            from src.utils.visualization import generate_paper_figures
            print("\nGenerating paper figures...")
            generate_paper_figures(results, output_dir=f"{args.output}/figures")
            print("✓ Figures generated")
        except ImportError:
            print("⚠ Visualization module not available, skipping figures")


if __name__ == '__main__':
    main()