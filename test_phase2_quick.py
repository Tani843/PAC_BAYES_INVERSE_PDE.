#!/usr/bin/env python3
"""
Quick Phase 2 Integration Test
Fast canary test to verify Phase 2 is working with PAC-Bayes pipeline
"""

import numpy as np
import sys
sys.path.append('.')

from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

def quick_test():
    """Quick test with simplified posterior."""
    
    print("=" * 60)
    print("QUICK PHASE 2 INTEGRATION TEST")
    print("=" * 60)
    
    # Simple PAC-Bayes-like posterior
    class QuickPosterior:
        def __init__(self, m=3):
            self.prior = self
            self.m = m
            self.kappa_min = 0.1
            self.kappa_max = 5.0
            # Simulate data
            self.y_obs = np.array([1.2, 2.1, 1.8])
            self.lambda_val = 1.0
            self.sigma = 0.1
            
        def sample(self, n, seed=None):
            rng = np.random.RandomState(seed)
            return rng.uniform(self.kappa_min, self.kappa_max, (n, self.m))
            
        def log_posterior(self, kappa):
            if np.any(kappa < self.kappa_min) or np.any(kappa > self.kappa_max):
                return -np.inf
            
            # Simple forward model: y = κ + noise
            y_pred = kappa
            
            # Likelihood term
            likelihood = -0.5 * np.sum((self.y_obs - y_pred)**2) / self.sigma**2
            
            # Prior (uniform)
            prior = 0.0
            
            # Gibbs posterior
            return self.lambda_val * likelihood + prior
    
    # Test Phase 2 sampler
    posterior = QuickPosterior(m=3)
    
    print("Setting up Phase 2 sampler...")
    sampler = AdaptiveMetropolisHastingsPhase2(
        posterior=posterior,
        initial_scale=0.05,
        seed=42,
        ess_target=100,     # Low target for quick test
        chunk_size=1000,    # Small chunks
        max_steps=5000,     # Quick limit
        use_block_updates=True
    )
    
    print("Running adaptive sampling...")
    try:
        result = sampler.run_adaptive_length(
            kappa_init=np.array([1.0, 2.0, 1.5]),
            n_burn=500
        )
        
        print(f"\nRESULTS:")
        print(f"  Samples: {len(result['samples'])}")
        print(f"  ESS min: {np.min(result['final_ess']):.1f}")
        print(f"  ESS mean: {np.mean(result['final_ess']):.1f}")
        print(f"  Acceptance: {result.get('overall_acceptance_rate', 0):.3f}")
        print(f"  Converged: {result['converged']}")
        print(f"  Chunks: {result['n_chunks']}")
        
        # Check sample quality
        samples = result['samples']
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)
        
        print(f"\nSAMPLE QUALITY:")
        print(f"  Posterior means: {means}")
        print(f"  True values:     {posterior.y_obs}")
        print(f"  Posterior stds:  {stds}")
        
        # Success criteria
        success_criteria = [
            result.get('overall_acceptance_rate', 0) > 0.15,
            np.min(result['final_ess']) > 30,
            len(result['samples']) > 0,
            not np.any(np.isnan(means))
        ]
        
        success = all(success_criteria)
        
        print(f"\nSUCCESS CRITERIA:")
        print(f"  ✓ Acceptance > 0.15: {success_criteria[0]}")
        print(f"  ✓ ESS > 30: {success_criteria[1]}")
        print(f"  ✓ Samples generated: {success_criteria[2]}")
        print(f"  ✓ Valid means: {success_criteria[3]}")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick integration test."""
    
    success = quick_test()
    
    print(f"\n" + "="*60)
    if success:
        print("✅ PHASE 2 INTEGRATION SUCCESS!")
        print("   Ready for full PAC-Bayes pipeline integration")
        print("   Proceed with comprehensive canary tests")
    else:
        print("❌ PHASE 2 INTEGRATION FAILED")
        print("   Debug issues before proceeding")
    print("="*60)
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)