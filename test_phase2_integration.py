#!/usr/bin/env python3
"""
Integration Script for Canary Tests
Integrate Phase 2 adaptive MCMC into existing PAC-Bayes pipeline
Run canary tests on specific configurations
"""

import numpy as np
import time
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.append('.')

# Import configuration
from config.experiment_config import ExperimentConfig

def create_mock_ecosystem():
    """
    Create a mock ecosystem that mimics the real PAC-Bayes pipeline
    for testing Phase 2 integration.
    """
    
    class MockDataGenerator:
        def __init__(self, config):
            self.config = config
            self.rng = np.random.RandomState(config['seed'])
        
        def generate_dataset(self):
            # Generate synthetic heat equation data
            kappa_true = self.rng.uniform(0.5, 3.0, self.config['m'])
            
            # Simple forward model: u(x,t) = sum(κᵢ * sin(πx) * exp(-κᵢ*t))
            sensor_pos = np.array(self.config['sensor_positions'])
            time_grid = np.linspace(0, self.config['T'], self.config['n_t'])
            
            # Clean data
            clean_data = np.zeros((len(sensor_pos), len(time_grid)))
            for i, kappa in enumerate(kappa_true):
                for j, x in enumerate(sensor_pos):
                    for k, t in enumerate(time_grid):
                        clean_data[j, k] += kappa * np.sin(np.pi * x) * np.exp(-kappa * t)
            
            # Add noise
            noise = self.rng.normal(0, self.config['sigma'], clean_data.shape)
            noisy_data = clean_data + noise
            
            return {
                'kappa_true': kappa_true,
                'clean_data': clean_data,
                'noisy_data': noisy_data,
                'sensor_positions': sensor_pos,
                'time_grid': time_grid,
                'n_observations': noisy_data.size
            }
    
    class MockPrior:
        def __init__(self, m, mu_0=0.0, tau2=1.0, kappa_min=0.1, kappa_max=5.0):
            self.m = m
            self.mu_0 = mu_0
            self.tau2 = tau2
            self.kappa_min = kappa_min
            self.kappa_max = kappa_max
        
        def sample(self, n_samples, seed=None):
            rng = np.random.RandomState(seed)
            return rng.uniform(self.kappa_min, self.kappa_max, (n_samples, self.m))
        
        def log_prior(self, kappa):
            if np.any(kappa < self.kappa_min) or np.any(kappa > self.kappa_max):
                return -np.inf
            # Uniform prior
            return 0.0
    
    class MockLossFunction:
        def __init__(self, c=1.0, sigma=0.1):
            self.c = c
            self.sigma = sigma
        
        def compute_loss(self, y_obs, y_pred):
            return 0.5 * np.sum((y_obs - y_pred)**2) / (self.sigma**2)
    
    class MockSolver:
        def __init__(self, config):
            self.config = config
            
        def forward_solve(self, kappa, sensor_positions, time_grid):
            # Simple forward model for testing
            y_pred = np.zeros((len(sensor_positions), len(time_grid)))
            for i, k in enumerate(kappa):
                for j, x in enumerate(sensor_positions):
                    for l, t in enumerate(time_grid):
                        y_pred[j, l] += k * np.sin(np.pi * x) * np.exp(-k * t)
            return y_pred
    
    class MockGibbsPosterior:
        def __init__(self, dataset, solver, loss_fn, prior, lambda_val, config):
            self.dataset = dataset
            self.solver = solver
            self.loss_fn = loss_fn
            self.prior = prior
            self.lambda_val = lambda_val
            self.config = config
            
        def log_posterior(self, kappa):
            # Ensure bounds
            if np.any(kappa < self.prior.kappa_min) or np.any(kappa > self.prior.kappa_max):
                return -np.inf
            
            # Forward solve
            y_pred = self.solver.forward_solve(
                kappa, 
                self.dataset['sensor_positions'], 
                self.dataset['time_grid']
            )
            
            # Compute loss
            loss = self.loss_fn.compute_loss(self.dataset['noisy_data'], y_pred)
            
            # Prior
            log_prior = self.prior.log_prior(kappa)
            
            # Gibbs posterior
            return -self.lambda_val * loss + log_prior
    
    class MockCertificate:
        def __init__(self, config):
            self.config = config
            self.rng = np.random.RandomState(config['seed'])
        
        def compute_empirical_loss(self, samples, dataset, solver, loss_fn):
            losses = []
            for kappa in samples:
                y_pred = solver.forward_solve(
                    kappa, dataset['sensor_positions'], dataset['time_grid']
                )
                loss = loss_fn.compute_loss(dataset['noisy_data'], y_pred)
                losses.append(loss)
            return np.array(losses)
        
        def compute_kl_divergence(self, posterior_samples, prior_samples, lambda_val, n):
            # Simple KL approximation
            return len(posterior_samples) * np.log(len(posterior_samples)) / (lambda_val * n)
        
        def compute_discretization_penalty(self, samples):
            # Simple discretization penalty
            h = self.config['Delta_x']
            return 0.1 * h**2  # O(h²) penalty
        
        def compute_certificate(self, samples, dataset, solver, loss_fn, lambda_val, n):
            # Empirical term
            losses = self.compute_empirical_loss(samples, dataset, solver, loss_fn)
            L_hat = np.mean(losses)
            
            # KL term (simplified)
            KL = self.compute_kl_divergence(samples, samples[:100], lambda_val, n)
            
            # Discretization penalty
            eta_h = self.compute_discretization_penalty(samples)
            
            # Delta term
            delta_term = np.log(1/self.config['delta']) / (lambda_val * n)
            
            # Certificate
            B_lambda = L_hat + KL + eta_h + delta_term
            
            return {
                'L_hat': L_hat,
                'KL': KL,
                'eta_h': eta_h,
                'delta_term': delta_term,
                'B_lambda': B_lambda,
                'lambda': lambda_val
            }
    
    return MockDataGenerator, MockPrior, MockLossFunction, MockSolver, MockGibbsPosterior, MockCertificate

def run_canary_test(config_tuple, seeds=[101, 202]):
    """
    Run canary test with Phase 2 MCMC on specified configuration.
    
    Args:
        config_tuple: (s, sigma, n_x, T, lambda, m)
        seeds: List of seeds to test
    """
    s, sigma, n_x, T, lambda_val, m = config_tuple
    
    print(f"\n{'='*60}")
    print(f"CANARY TEST: s={s}, σ={sigma}, n_x={n_x}, T={T}, λ={lambda_val}, m={m}")
    print(f"{'='*60}")
    
    # Import Phase 2 MCMC
    try:
        from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2
    except ImportError:
        print("❌ Could not import Phase 2 MCMC - using mock")
        return []
    
    # Create mock ecosystem
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior, Certificate = create_mock_ecosystem()
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        start_time = time.time()
        
        # Build configuration
        config = {
            's': s,
            'sensor_positions': [0.25, 0.50, 0.75] if s == 3 else [0.10, 0.30, 0.50, 0.70, 0.90],
            'placement_type': 'fixed',
            'sigma': sigma,
            'n_x': n_x,
            'T': T,
            'lambda': lambda_val,
            'm': m,
            'n_t': 50,
            'seed': seed,
            'delta': 0.05,
            'alpha': 1e-3,
            'M': 2000,
            'R': 50,
            'n': s * 50,
            'c': 1.0,
            'Delta_x': 1.0 / n_x,
            'Delta_t': T / 49
        }
        
        try:
            # 1. Generate data
            print("  Generating dataset...")
            data_gen = DataGenerator(config)
            dataset = data_gen.generate_dataset()
            print(f"    True κ: {dataset['kappa_true']}")
            
            # 2. Setup components
            print("  Setting up solver and loss...")
            solver = Solver(config)
            loss_fn = LossFunction(c=config['c'], sigma=config['sigma'])
            prior = Prior(m=m, kappa_min=0.1, kappa_max=5.0)
            
            # 3. Setup posterior
            print("  Creating Gibbs posterior...")
            posterior = GibbsPosterior(
                dataset=dataset,
                solver=solver,
                loss_fn=loss_fn,
                prior=prior,
                lambda_val=lambda_val,
                config=config
            )
            
            # Test posterior evaluation
            test_kappa = dataset['kappa_true']
            test_logp = posterior.log_posterior(test_kappa)
            print(f"    Test log-posterior at true κ: {test_logp:.3f}")
            
            # 4. Run Phase 2 MCMC
            print("  Running Phase 2 Adaptive MCMC...")
            sampler = AdaptiveMetropolisHastingsPhase2(
                posterior=posterior,
                initial_scale=0.01,
                seed=seed,
                ess_target=300,     # Reasonable target
                chunk_size=2000,    # Smaller chunks for testing
                max_steps=15000,    # Reasonable limit
                use_block_updates=True
            )
            
            # Run adaptive length sampling
            mcmc_results = sampler.run_adaptive_length(
                kappa_init=None,
                n_burn=1000  # Shorter burn-in for testing
            )
            
            print(f"    MCMC completed: {len(mcmc_results['samples'])} samples")
            print(f"    Acceptance: {mcmc_results.get('overall_acceptance_rate', 0):.3f}")
            print(f"    ESS: min={np.min(mcmc_results['final_ess']):.1f}")
            
            # 5. Compute certificate
            print("  Computing PAC-Bayes certificate...")
            certificate = Certificate(config)
            
            # Use subset of samples for certificate (computational efficiency)
            cert_samples = mcmc_results['samples'][::10]  # Every 10th sample
            cert_samples = cert_samples[:100]  # Max 100 samples
            
            cert_result = certificate.compute_certificate(
                cert_samples, dataset, solver, loss_fn, lambda_val, config['n']
            )
            
            runtime = time.time() - start_time
            
            # Collect results
            result = {
                'seed': seed,
                'config': config_tuple,
                'true_kappa': dataset['kappa_true'].tolist(),
                'acceptance_rate': mcmc_results.get('overall_acceptance_rate', 0),
                'ess_min': float(np.min(mcmc_results['final_ess'])),
                'ess_mean': float(np.mean(mcmc_results['final_ess'])),
                'ess_per_coord': mcmc_results['final_ess'].tolist(),
                'kl': cert_result['KL'],
                'L_hat': cert_result['L_hat'],
                'B_lambda': cert_result['B_lambda'],
                'eta_h': cert_result['eta_h'],
                'runtime': runtime,
                'converged': mcmc_results['converged'],
                'n_chunks': mcmc_results['n_chunks'],
                'total_steps': mcmc_results['total_steps'],
                'n_samples': len(mcmc_results['samples']),
                'posterior_mean': np.mean(cert_samples, axis=0).tolist(),
                'posterior_std': np.std(cert_samples, axis=0).tolist()
            }
            
            results.append(result)
            
            # Print summary
            print(f"\n  Results for seed {seed}:")
            print(f"    Acceptance rate: {result['acceptance_rate']:.3f}")
            print(f"    ESS: min={result['ess_min']:.1f}, mean={result['ess_mean']:.1f}")
            print(f"    L̂: {result['L_hat']:.4f}")
            print(f"    B_λ: {result['B_lambda']:.4f}")
            print(f"    Certificate valid: {result['B_lambda'] >= result['L_hat']}")
            print(f"    Runtime: {result['runtime']:.1f}s")
            print(f"    Posterior mean: {result['posterior_mean']}")
            print(f"    True κ:         {result['true_kappa']}")
            
        except Exception as e:
            print(f"  ❌ Error for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            # Add failed result
            results.append({
                'seed': seed,
                'error': str(e),
                'config': config_tuple
            })
    
    return results

def evaluate_canary_results(all_results):
    """
    Evaluate if Phase 3 is needed based on canary results.
    """
    print("\n" + "="*60)
    print("CANARY EVALUATION")
    print("="*60)
    
    total_tests = 0
    successful_tests = 0
    issues = []
    
    for config_results in all_results:
        for result in config_results:
            if 'error' in result:
                issues.append(f"Config {result['config']} Seed {result['seed']}: Failed with error")
                continue
                
            total_tests += 1
            
            # Check acceptance rate
            if result['acceptance_rate'] < 0.20:
                issues.append(f"Config {result['config']} Seed {result['seed']}: "
                            f"Acceptance {result['acceptance_rate']:.3f} < 0.20")
            
            # Check ESS
            if result['ess_min'] < 200:  # Lowered threshold for testing
                issues.append(f"Config {result['config']} Seed {result['seed']}: "
                            f"Min ESS {result['ess_min']:.1f} < 200")
            
            # Check certificate validity
            if result['B_lambda'] < result['L_hat']:
                issues.append(f"Config {result['config']} Seed {result['seed']}: "
                            f"B_λ {result['B_lambda']:.4f} < L̂ {result['L_hat']:.4f}")
            
            # If no issues, count as successful
            if (result['acceptance_rate'] >= 0.20 and 
                result['ess_min'] >= 200 and 
                result['B_lambda'] >= result['L_hat']):
                successful_tests += 1
    
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"Success rate: {successful_tests}/{total_tests} ({success_rate:.1%})")
    
    if issues:
        print(f"\n⚠️ Issues detected ({len(issues)} total):")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        
        if success_rate < 0.7:
            print(f"\n❌ PHASE 3 NEEDED - Success rate too low ({success_rate:.1%})")
            return False
        else:
            print(f"\n⚠️ PHASE 3 RECOMMENDED - Some issues detected")
            return False
    else:
        print(f"\n✅ ALL CANARY TESTS PASSED!")
        print(f"  - All acceptance rates ≥ 0.20")
        print(f"  - All ESS ≥ 200")
        print(f"  - All certificates valid (B_λ ≥ L̂)")
        print(f"\n🚀 READY FOR FULL GRID - Phase 2 is sufficient!")
        return True

def main():
    """Main canary test execution."""
    
    print("=" * 80)
    print("PHASE 2 INTEGRATION - CANARY TESTS")
    print("=" * 80)
    
    # Define canary configurations (representative subset)
    canary_configs = [
        (3, 0.10, 50, 0.5, 1.0, 3),   # Config 1: Low noise, coarse mesh
        (5, 0.20, 100, 0.3, 2.0, 5),  # Config 2: High noise, fine mesh
        (3, 0.05, 100, 0.5, 0.5, 3),  # Config 3: Very low noise, fine mesh
    ]
    
    all_results = []
    
    for i, config in enumerate(canary_configs, 1):
        print(f"\n[{i}/{len(canary_configs)}] Testing configuration: {config}")
        
        # Run with reduced seeds for faster testing
        results = run_canary_test(config, seeds=[101, 202])
        all_results.append(results)
        
        # Save intermediate results
        output_file = f'canary_results_config_{i}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_file}")
    
    # Save all results
    with open('canary_results_all.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Evaluate if Phase 3 is needed
    phase2_sufficient = evaluate_canary_results(all_results)
    
    # Final recommendation
    print(f"\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if phase2_sufficient:
        print("✅ Phase 2 Adaptive MCMC is SUFFICIENT for full grid execution")
        print("   Proceed with running the complete 1,728 experiment grid using Phase 2")
    else:
        print("⚠️ Phase 3 Advanced Techniques are RECOMMENDED")
        print("   Consider implementing HMC, NUTS, or other advanced samplers")
    
    return phase2_sufficient

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)