#!/usr/bin/env python3
"""
Quick test of Phase 2 MCMC with reduced parameters
"""

import json
import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import traceback

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Add project root to path
sys.path.append('.')

# Import configuration
from config.experiment_config import ExperimentConfig

# Import Phase 2 MCMC
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

def create_quick_test_ecosystem():
    """
    Create simplified ecosystem for quick testing.
    """
    
    class QuickDataGenerator:
        """Simplified data generator."""
        def __init__(self, config):
            self.config = config
            self.rng = np.random.RandomState(config['seed'])
        
        def generate_dataset(self):
            m = self.config['m']
            s = self.config['s']
            
            # Simple true parameter
            kappa_true = self.rng.uniform(1.0, 3.0, m)
            
            # Simple sensor data (m x s matrix)
            x_sensors = np.array(self.config['sensor_positions'])
            n_t = 20  # Reduced time points
            t_grid = np.linspace(0, self.config['T'], n_t)
            
            # Simple heat equation approximation
            clean_data = np.zeros((s, n_t))
            for i, x in enumerate(x_sensors):
                for j, t in enumerate(t_grid):
                    # Simplified response
                    u_val = 0.0
                    for k, kappa_k in enumerate(kappa_true):
                        segment_center = (k + 0.5) / m
                        weight = np.exp(-2 * (x - segment_center)**2)
                        u_val += weight * kappa_k * np.exp(-0.5 * kappa_k * t)
                    clean_data[i, j] = u_val
            
            # Add noise
            noise = self.rng.normal(0, self.config['sigma'], clean_data.shape)
            noisy_data = clean_data + noise
            
            return {
                'kappa_star': kappa_true,
                'clean_data': clean_data,
                'noisy_data': noisy_data,
                'sensor_positions': x_sensors,
                'time_grid': t_grid,
                'n_observations': noisy_data.size
            }
    
    class QuickPrior:
        """Simplified prior."""
        def __init__(self, m, mu_0=0.0, tau2=1.0, kappa_min=0.1, kappa_max=5.0):
            self.m = m
            self.mu_0 = mu_0
            self.tau2 = tau2
            self.kappa_min = kappa_min
            self.kappa_max = kappa_max
        
        def sample(self, n_samples, seed=None):
            """Simple uniform sampling in bounds."""
            rng = np.random.RandomState(seed)
            return rng.uniform(self.kappa_min, self.kappa_max, (n_samples, self.m))
        
        def log_prior(self, kappa):
            """Uniform log-prior."""
            if np.any(kappa < self.kappa_min) or np.any(kappa > self.kappa_max):
                return -np.inf
            return 0.0  # Uniform density (unnormalized)
    
    class QuickLoss:
        """Simplified loss function."""
        def __init__(self, c=1.0, sigma=0.1):
            self.c = c
            self.sigma = sigma
        
        def compute_loss(self, y_obs, y_pred):
            # Simple squared error, bounded
            mse = np.mean((y_obs - y_pred)**2)
            return min(mse / (self.c * self.sigma**2), 1.0)
    
    class QuickSolver:
        """Simplified forward solver."""
        def __init__(self, config):
            self.config = config
            
        def forward_solve(self, kappa, sensor_positions, time_grid):
            """Simple forward model."""
            s = len(sensor_positions)
            n_t = len(time_grid)
            m = len(kappa)
            
            y_pred = np.zeros((s, n_t))
            
            for i, x in enumerate(sensor_positions):
                for j, t in enumerate(time_grid):
                    u_val = 0.0
                    for k in range(m):
                        segment_center = (k + 0.5) / m
                        weight = np.exp(-2 * (x - segment_center)**2)
                        u_val += weight * kappa[k] * np.exp(-0.5 * kappa[k] * t)
                    y_pred[i, j] = u_val
            
            return y_pred
    
    class QuickGibbsPosterior:
        """Simplified Gibbs posterior."""
        def __init__(self, dataset, solver, loss_fn, prior, lambda_val, config):
            self.dataset = dataset
            self.solver = solver
            self.loss_fn = loss_fn
            self.prior = prior
            self.lambda_val = lambda_val
            self.config = config
            
        def log_posterior(self, kappa):
            # Check bounds
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
            
            # Prior term
            log_prior = self.prior.log_prior(kappa)
            
            # Simplified Gibbs posterior
            return -self.lambda_val * loss + log_prior
    
    return QuickDataGenerator, QuickPrior, QuickLoss, QuickSolver, QuickGibbsPosterior

def test_phase2_quick():
    """
    Quick test of Phase 2 MCMC with a single simple configuration.
    """
    
    print("=" * 60)
    print("QUICK TEST: PHASE 2 ADAPTIVE MCMC")
    print("=" * 60)
    
    # Simple test configuration
    test_config = {
        's': 3,
        'sensor_positions': [0.25, 0.50, 0.75],
        'sigma': 0.1,
        'lambda': 1.0,
        'm': 3,
        'T': 0.5,
        'seed': 101,
        'n': 60  # 3 sensors x 20 time points
    }
    
    print(f"Test config: s={test_config['s']}, m={test_config['m']}, λ={test_config['lambda']}")
    
    # Create ecosystem
    DataGenerator, Prior, LossFunction, Solver, GibbsPosterior = create_quick_test_ecosystem()
    
    start_time = time.time()
    
    try:
        # 1. Generate data
        print(f"[1/4] Generating dataset...")
        data_gen = DataGenerator(test_config)
        dataset = data_gen.generate_dataset()
        print(f"  Dataset: {dataset['n_observations']} observations")
        print(f"  True κ: {dataset['kappa_star']}")
        
        # 2. Setup components
        print(f"[2/4] Setting up solver and posterior...")
        solver = Solver(test_config)
        loss_fn = LossFunction(c=1.0, sigma=test_config['sigma'])
        prior = Prior(m=test_config['m'], kappa_min=0.1, kappa_max=5.0)
        
        posterior = GibbsPosterior(
            dataset=dataset,
            solver=solver,
            loss_fn=loss_fn,
            prior=prior,
            lambda_val=test_config['lambda'],
            config=test_config
        )
        
        # Test posterior evaluation
        test_kappa = dataset['kappa_star']
        test_logp = posterior.log_posterior(test_kappa)
        print(f"  Log-posterior at true κ: {test_logp:.2f}")
        
        # 3. Run Phase 2 MCMC (with reduced parameters)
        print(f"[3/4] Running Phase 2 MCMC...")
        sampler = AdaptiveMetropolisHastingsPhase2(
            posterior=posterior,
            initial_scale=0.05,  # More conservative
            seed=test_config['seed'],
            ess_target=50,       # Reduced target
            chunk_size=500,      # Smaller chunks
            max_steps=5000,      # Reduced max steps
            use_block_updates=True
        )
        
        mcmc_results = sampler.run_adaptive_length(n_burn=200)  # Reduced burn-in
        
        print(f"  MCMC completed:")
        print(f"    Total steps: {mcmc_results['total_steps']}")
        print(f"    Chunks: {mcmc_results['n_chunks']}")
        print(f"    Min ESS: {np.min(mcmc_results['final_ess']):.1f}")
        print(f"    Mean ESS: {np.mean(mcmc_results['final_ess']):.1f}")
        print(f"    Converged: {mcmc_results['converged']}")
        print(f"    Blocks used: {mcmc_results['blocks_used']}")
        
        # 4. Basic certificate computation
        print(f"[4/4] Computing basic certificate...")
        samples = mcmc_results['samples']
        
        # Compute empirical loss on samples
        empirical_losses = []
        for kappa in samples[::10][:50]:  # Subsample for speed
            y_pred = solver.forward_solve(
                kappa, dataset['sensor_positions'], dataset['time_grid']
            )
            loss = loss_fn.compute_loss(dataset['noisy_data'], y_pred)
            empirical_losses.append(loss)
        
        L_hat = np.mean(empirical_losses)
        
        # Basic certificate bound (simplified)
        basic_bound = L_hat + 0.1  # Simplified
        
        print(f"  Empirical loss L̂: {L_hat:.4f}")
        print(f"  Basic bound: {basic_bound:.4f}")
        
        runtime = time.time() - start_time
        print(f"\nTest completed successfully in {runtime:.1f}s")
        
        # Summary
        result = {
            'test_config': test_config,
            'mcmc_summary': {
                'total_steps': mcmc_results['total_steps'],
                'min_ess': float(np.min(mcmc_results['final_ess'])),
                'converged': mcmc_results['converged'],
                'blocks_used': mcmc_results['blocks_used']
            },
            'certificate_summary': {
                'empirical_loss': L_hat,
                'basic_bound': basic_bound
            },
            'runtime': runtime,
            'success': True
        }
        
        return result
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        
        return {
            'test_config': test_config,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'runtime': runtime,
            'success': False
        }

def main():
    """Main test function."""
    
    try:
        result = test_phase2_quick()
        
        # Save result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'test_phase2_quick_{timestamp}.json'
        
        safe_result = make_json_serializable(result)
        with open(output_file, 'w') as f:
            json.dump(safe_result, f, indent=2)
        
        if result['success']:
            print(f"\n✅ Phase 2 MCMC test successful!")
            print(f"   Results saved to: {output_file}")
            return True
        else:
            print(f"\n❌ Phase 2 MCMC test failed!")
            print(f"   Error details saved to: {output_file}")
            return False
        
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)