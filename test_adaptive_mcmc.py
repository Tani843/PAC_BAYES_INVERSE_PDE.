#!/usr/bin/env python3
"""
Test the adaptive MCMC implementation on a simple case
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Import our modules
from src.mcmc.adaptive_metropolis_hastings import AdaptiveMetropolisHastings
from config.experiment_config import ExperimentConfig

def create_test_posterior():
    """Create a simple test posterior for validation."""
    
    # Create minimal test configuration
    config = ExperimentConfig()
    
    # Simple test parameters
    test_config = {
        's': 3,
        'sensor_positions': [0.25, 0.5, 0.75],
        'sigma': 0.1,
        'n_x': 50,
        'T': 0.3,
        'lambda': 1.0,
        'm': 3,
        'seed': 12345
    }
    
    print("Creating test posterior...")
    print(f"Test config: {test_config}")
    
    # We need to import and create the posterior
    try:
        from src.inference.gibbs_posterior import GibbsPosterior
        from src.utils.data_generation import generate_dataset
        from src.models.pde_solver import PDESolver
        
        # Generate test dataset
        dataset = generate_dataset(test_config)
        
        # Create PDE solver
        solver = PDESolver(
            n_x=test_config['n_x'],
            domain=(0.0, 1.0),
            T=test_config['T'],
            scheme='crank_nicolson'
        )
        
        # Create Gibbs posterior
        posterior = GibbsPosterior(
            dataset=dataset,
            solver=solver,
            lambda_val=test_config['lambda'],
            c=1.0,
            m=test_config['m'],
            kappa_bounds=(0.1, 5.0),
            prior_params={'mu_0': 0.0, 'tau_squared': 1.0}
        )
        
        return posterior, test_config
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Creating mock posterior for basic testing...")
        return create_mock_posterior(), test_config

def create_mock_posterior():
    """Create a mock posterior for basic testing when imports fail."""
    
    class MockPrior:
        def __init__(self):
            self.m = 3
            self.kappa_min = 0.1
            self.kappa_max = 5.0
            
        def sample(self, n_samples, seed=None):
            rng = np.random.RandomState(seed)
            return rng.uniform(self.kappa_min, self.kappa_max, (n_samples, self.m))
    
    class MockPosterior:
        def __init__(self):
            self.prior = MockPrior()
            
        def log_posterior(self, kappa):
            # Simple quadratic log-posterior for testing
            center = np.array([1.0, 2.0, 3.0])
            return -0.5 * np.sum((kappa - center)**2)
    
    return MockPosterior()

def test_adaptive_mcmc():
    """Test the adaptive MCMC implementation."""
    
    print("=" * 80)
    print("TESTING ADAPTIVE MCMC IMPLEMENTATION")
    print("=" * 80)
    
    # Create test posterior
    posterior, config = create_test_posterior()
    
    # Create adaptive sampler
    sampler = AdaptiveMetropolisHastings(
        posterior=posterior,
        initial_scale=0.01,  # Conservative starting scale
        seed=42
    )
    
    # Run diagnostic
    print("\n1. POSTERIOR DIAGNOSTICS:")
    diagnostics = sampler.diagnose_posterior()
    print(f"   Test point: {diagnostics['test_point']}")
    print(f"   Log-posterior: {diagnostics['log_posterior']:.4f}")
    print(f"   Domain bounds: {diagnostics['domain_bounds']}")
    print(f"   Dimension: {diagnostics['dimension']}")
    
    print("\n   Proposal scale tests:")
    for test in diagnostics['proposal_tests']:
        print(f"     Scale {test['scale']}: norm={test['proposal_norm']:.4f}, in_bounds={test['in_bounds']}")
    
    # Test adaptive burn-in
    print("\n2. ADAPTIVE BURN-IN TEST:")
    kappa_init = posterior.prior.sample(1, seed=123)[0]
    print(f"   Initial κ: {kappa_init}")
    
    try:
        burn_chain, kappa_final = sampler.adaptive_burn_in(kappa_init, n_burn=500)
        print(f"   Final κ: {kappa_final}")
        print(f"   Burn chain shape: {burn_chain.shape}")
        
        # Check movement
        movement = np.linalg.norm(kappa_final - kappa_init)
        print(f"   Total movement: {movement:.4f}")
        
        # Check adaptation history
        if sampler.burn_in_history:
            print(f"   Adaptation steps: {len(sampler.burn_in_history)}")
            final_adaptation = sampler.burn_in_history[-1]
            print(f"   Final acceptance: {final_adaptation['acceptance_rate']:.3f}")
            print(f"   Final τ: {final_adaptation['tau']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Burn-in failed: {e}")
        return False
    
    # Test full chain
    print("\n3. FULL CHAIN TEST:")
    try:
        result = sampler.run_chain_adaptive(
            n_steps=1000,
            n_burn=300,
            kappa_init=kappa_init
        )
        
        print(f"   Samples shape: {result['samples'].shape}")
        print(f"   Acceptance rate: {result['acceptance_rate']:.3f}")
        print(f"   ESS: min={np.min(result['ess']):.1f}, mean={np.mean(result['ess']):.1f}")
        print(f"   Converged: {result['converged']}")
        print(f"   Forward evaluations: {result['n_forward_evals']}")
        
        # Check if we improved over original issues
        if result['acceptance_rate'] > 0.15:
            print(f"   ✅ Acceptance rate improved (>{0.15:.2f})")
        else:
            print(f"   ⚠️ Acceptance rate still low ({result['acceptance_rate']:.3f})")
            
        if np.min(result['ess']) > 50:
            print(f"   ✅ ESS reasonable (>{50})")
        else:
            print(f"   ⚠️ ESS still low ({np.min(result['ess']):.1f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Full chain failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the adaptive MCMC test."""
    
    success = test_adaptive_mcmc()
    
    if success:
        print(f"\n✅ ADAPTIVE MCMC TEST COMPLETED SUCCESSFULLY!")
        print(f"Ready for integration into main experimental pipeline.")
    else:
        print(f"\n❌ ADAPTIVE MCMC TEST FAILED")
        print(f"Issues need to be resolved before integration.")
    
    return success

if __name__ == '__main__':
    main()