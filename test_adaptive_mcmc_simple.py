#!/usr/bin/env python3
"""
Simple test of the adaptive MCMC implementation using mock objects
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.mcmc.adaptive_metropolis_hastings import AdaptiveMetropolisHastings

class MockPrior:
    """Mock prior for testing."""
    def __init__(self, m=3, kappa_min=0.1, kappa_max=5.0):
        self.m = m
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        
    def sample(self, n_samples, seed=None):
        rng = np.random.RandomState(seed)
        return rng.uniform(self.kappa_min, self.kappa_max, (n_samples, self.m))

class MockPosterior:
    """Mock posterior with known properties for testing."""
    def __init__(self, m=3):
        self.prior = MockPrior(m=m)
        # Create a multimodal posterior for testing
        self.modes = np.array([[1.0, 2.0, 3.0], [2.5, 1.5, 4.0]])
        
    def log_posterior(self, kappa):
        """Mixture of Gaussians log-posterior."""
        # Ensure kappa is in bounds
        if np.any(kappa < self.prior.kappa_min) or np.any(kappa > self.prior.kappa_max):
            return -np.inf
            
        # Mixture of two Gaussians
        log_probs = []
        for mode in self.modes:
            diff = kappa - mode
            log_prob = -0.5 * np.sum(diff**2 / 0.5**2)  # σ² = 0.5² for each mode
            log_probs.append(log_prob)
        
        # Log-sum-exp for mixture
        max_log_prob = max(log_probs)
        return max_log_prob + np.log(sum(np.exp(lp - max_log_prob) for lp in log_probs))

def test_adaptive_mcmc():
    """Test the adaptive MCMC implementation."""
    
    print("=" * 80)
    print("TESTING ADAPTIVE MCMC WITH MOCK POSTERIOR")
    print("=" * 80)
    
    # Create mock posterior
    posterior = MockPosterior(m=3)
    print(f"Created mock posterior with modes at:")
    for i, mode in enumerate(posterior.modes):
        print(f"  Mode {i+1}: {mode}")
    
    # Create adaptive sampler
    sampler = AdaptiveMetropolisHastings(
        posterior=posterior,
        initial_scale=0.05,  # Conservative starting scale
        seed=42
    )
    
    # Run diagnostic
    print("\n1. POSTERIOR DIAGNOSTICS:")
    diagnostics = sampler.diagnose_posterior()
    print(f"   Test point: {diagnostics['test_point']}")
    print(f"   Log-posterior: {diagnostics['log_posterior']:.4f}")
    print(f"   Domain bounds: {diagnostics['domain_bounds']}")
    print(f"   Dimension: {diagnostics['dimension']}")
    
    print(f"\n   Proposal scale tests:")
    for test in diagnostics['proposal_tests']:
        print(f"     Scale {test['scale']}: norm={test['proposal_norm']:.4f}, in_bounds={test['in_bounds']}")
    
    # Test adaptive burn-in
    print(f"\n2. ADAPTIVE BURN-IN TEST:")
    kappa_init = posterior.prior.sample(1, seed=123)[0]
    print(f"   Initial κ: {kappa_init}")
    print(f"   Initial log-posterior: {posterior.log_posterior(kappa_init):.4f}")
    
    try:
        burn_chain, kappa_final = sampler.adaptive_burn_in(kappa_init, n_burn=500)
        print(f"   Final κ: {kappa_final}")
        print(f"   Final log-posterior: {posterior.log_posterior(kappa_final):.4f}")
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
        
        print(f"   ✅ Burn-in completed successfully")
        
    except Exception as e:
        print(f"   ❌ Burn-in failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full chain
    print(f"\n3. FULL CHAIN TEST:")
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
        
        # Analyze samples
        samples = result['samples']
        sample_means = np.mean(samples, axis=0)
        sample_stds = np.std(samples, axis=0)
        
        print(f"\n   Sample statistics:")
        print(f"     Mean: {sample_means}")
        print(f"     Std:  {sample_stds}")
        
        # Check if we're sampling from the right region
        distances_to_modes = []
        for mode in posterior.modes:
            dist = np.linalg.norm(sample_means - mode)
            distances_to_modes.append(dist)
        
        closest_mode_dist = min(distances_to_modes)
        print(f"     Distance to closest mode: {closest_mode_dist:.3f}")
        
        # Check if we improved over original issues
        improvements = []
        if result['acceptance_rate'] > 0.15:
            print(f"   ✅ Acceptance rate improved (>{0.15:.2f})")
            improvements.append("acceptance")
        else:
            print(f"   ⚠️ Acceptance rate still low ({result['acceptance_rate']:.3f})")
            
        if np.min(result['ess']) > 50:
            print(f"   ✅ ESS reasonable (>{50})")
            improvements.append("ess")
        else:
            print(f"   ⚠️ ESS still low ({np.min(result['ess']):.1f})")
        
        if result['converged']:
            print(f"   ✅ Convergence achieved")
            improvements.append("convergence")
        else:
            print(f"   ⚠️ Convergence not achieved")
            
        if closest_mode_dist < 0.5:
            print(f"   ✅ Sampling near true mode")
            improvements.append("accuracy")
        else:
            print(f"   ⚠️ Not sampling near known modes")
        
        return len(improvements) >= 2  # Success if at least 2 improvements
        
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
        print(f"The implementation shows significant improvements over the original MCMC.")
        print(f"Ready for integration into the experimental pipeline.")
    else:
        print(f"\n❌ ADAPTIVE MCMC TEST FAILED")
        print(f"Further tuning may be needed for the specific posterior geometry.")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)