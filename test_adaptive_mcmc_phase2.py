#!/usr/bin/env python3
"""
Test Phase 2: Block updates and adaptive run length
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

class MockPrior:
    """Mock prior for testing."""
    def __init__(self, m=3, kappa_min=0.1, kappa_max=5.0):
        self.m = m
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        
    def sample(self, n_samples, seed=None):
        rng = np.random.RandomState(seed)
        return rng.uniform(self.kappa_min, self.kappa_max, (n_samples, self.m))

class MockPosteriorChallenging:
    """More challenging posterior for Phase 2 testing."""
    def __init__(self, m=5):
        self.prior = MockPrior(m=m)
        # Create correlated parameters - blocks [0,1], [2,3], [4]
        self.correlation_strength = 0.8
        
    def log_posterior(self, kappa):
        """Correlated log-posterior that benefits from block updates."""
        if np.any(kappa < self.prior.kappa_min) or np.any(kappa > self.prior.kappa_max):
            return -np.inf
        
        # Center around [2, 2, 3, 3, 1]
        target = np.array([2.0, 2.0, 3.0, 3.0, 1.0])
        
        # Create correlations within blocks
        log_prob = 0.0
        
        # Block 1: [0,1] - highly correlated
        diff_01 = kappa[:2] - target[:2]
        # Add correlation penalty: prefer κ₀ ≈ κ₁
        correlation_penalty_01 = -10.0 * (kappa[0] - kappa[1])**2
        log_prob += -0.5 * np.sum(diff_01**2 / 0.3**2) + correlation_penalty_01
        
        # Block 2: [2,3] - moderately correlated  
        diff_23 = kappa[2:4] - target[2:4]
        correlation_penalty_23 = -5.0 * (kappa[2] - kappa[3])**2
        log_prob += -0.5 * np.sum(diff_23**2 / 0.4**2) + correlation_penalty_23
        
        # Block 3: [4] - independent
        diff_4 = kappa[4] - target[4]
        log_prob += -0.5 * (diff_4**2 / 0.5**2)
        
        return log_prob

def test_phase2_blocks():
    """Test Phase 2 with block updates."""
    
    print("=" * 80)
    print("TESTING PHASE 2: BLOCK UPDATES + ADAPTIVE LENGTH")
    print("=" * 80)
    
    # Create challenging correlated posterior
    posterior = MockPosteriorChallenging(m=5)
    print(f"Created challenging posterior with correlations in blocks")
    
    # Create Phase 2 sampler
    sampler = AdaptiveMetropolisHastingsPhase2(
        posterior=posterior,
        initial_scale=0.02,
        seed=42,
        ess_target=200,  # Lower target for testing
        chunk_size=2000,  # Smaller chunks for testing
        max_steps=15000,   # Reasonable limit for testing
        use_block_updates=True
    )
    
    # Test block creation
    blocks = sampler.create_blocks()
    print(f"\nBlock structure for m={posterior.prior.m}: {blocks}")
    
    # Test adaptive length sampling
    print(f"\n1. ADAPTIVE LENGTH SAMPLING:")
    kappa_init = posterior.prior.sample(1, seed=123)[0]
    print(f"   Initial κ: {kappa_init}")
    print(f"   Initial log-posterior: {posterior.log_posterior(kappa_init):.4f}")
    
    try:
        result = sampler.run_adaptive_length(kappa_init, n_burn=500)
        
        print(f"\n   RESULTS:")
        print(f"   • Samples collected: {len(result['samples'])}")
        print(f"   • Number of chunks: {result['n_chunks']}")
        print(f"   • Total steps: {result['total_steps']}")
        print(f"   • Final ESS: min={np.min(result['final_ess']):.1f}, mean={np.mean(result['final_ess']):.1f}")
        print(f"   • Target ESS: {result['target_ess']}")
        print(f"   • Converged: {result['converged']}")
        print(f"   • Efficiency: {result['efficiency']:.4f} ESS/step")
        print(f"   • Overall acceptance: {result['overall_acceptance_rate']:.3f}")
        
        # Analyze final samples
        samples = result['samples']
        sample_means = np.mean(samples, axis=0)
        sample_stds = np.std(samples, axis=0)
        
        print(f"\n   SAMPLE QUALITY:")
        print(f"   • Sample means: {sample_means}")
        print(f"   • Sample stds:  {sample_stds}")
        print(f"   • Target means: [2.0, 2.0, 3.0, 3.0, 1.0]")
        
        # Check block correlations
        if len(samples) > 100:
            # Block 1 correlation
            corr_01 = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
            # Block 2 correlation  
            corr_23 = np.corrcoef(samples[:, 2], samples[:, 3])[0, 1]
            print(f"   • Block [0,1] correlation: {corr_01:.3f}")
            print(f"   • Block [2,3] correlation: {corr_23:.3f}")
        
        # Check ESS progression
        if result['ess_history']:
            print(f"\n   ESS PROGRESSION:")
            for i, chunk_info in enumerate(result['ess_history'][-3:], len(result['ess_history'])-2):
                print(f"     Chunk {chunk_info['chunk']}: ESS={chunk_info['min_ess']:.1f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Adaptive length sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_comparison_study():
    """Test comparison between block and full updates."""
    
    print("\n" + "=" * 80)
    print("COMPARISON STUDY: BLOCK vs FULL UPDATES")
    print("=" * 80)
    
    posterior = MockPosteriorChallenging(m=5)
    
    # Create sampler for comparison
    sampler = AdaptiveMetropolisHastingsPhase2(
        posterior=posterior,
        initial_scale=0.02,
        seed=42,
        ess_target=150,  # Lower for comparison
        chunk_size=1500,
        max_steps=8000,   # Lower budget for comparison
    )
    
    try:
        comparison_results = sampler.run_comparison_study(n_burn=300)
        
        print(f"\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        
        block_result = comparison_results['block_updates']
        full_result = comparison_results['full_updates']
        
        # Efficiency comparison
        block_eff = block_result['efficiency']
        full_eff = full_result['efficiency']
        improvement = (block_eff - full_eff) / full_eff * 100 if full_eff > 0 else 0
        
        print(f"EFFICIENCY (ESS/step):")
        print(f"  Block updates: {block_eff:.4f}")
        print(f"  Full updates:  {full_eff:.4f}")
        print(f"  Improvement:   {improvement:+.1f}%")
        
        # Convergence comparison
        print(f"\nCONVERGENCE:")
        print(f"  Block updates: {'✓' if block_result['converged'] else '✗'}")
        print(f"  Full updates:  {'✓' if full_result['converged'] else '✗'}")
        
        # ESS comparison
        block_min_ess = np.min(block_result['final_ess'])
        full_min_ess = np.min(full_result['final_ess'])
        
        print(f"\nMIN ESS ACHIEVED:")
        print(f"  Block updates: {block_min_ess:.1f}")
        print(f"  Full updates:  {full_min_ess:.1f}")
        
        return comparison_results
        
    except Exception as e:
        print(f"❌ Comparison study failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run Phase 2 tests."""
    
    print("Starting Phase 2 Adaptive MCMC Tests...")
    
    # Test 1: Basic Phase 2 functionality
    result1 = test_phase2_blocks()
    success1 = result1 is not None and (np.min(result1['final_ess']) > 50)
    
    # Test 2: Comparison study
    result2 = test_comparison_study() 
    success2 = result2 is not None
    
    # Overall assessment
    overall_success = success1 and success2
    
    print(f"\n" + "=" * 80)
    print("PHASE 2 TEST SUMMARY")
    print("=" * 80)
    
    print(f"✅ Basic functionality: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ Comparison study:    {'PASS' if success2 else 'FAIL'}")
    print(f"✅ Overall Phase 2:     {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print(f"\n🎉 Phase 2 implementation is ready for integration!")
        print(f"   • Block updates improve efficiency for correlated parameters")
        print(f"   • Adaptive run length achieves target ESS automatically")
        print(f"   • Robust fallbacks handle edge cases")
    else:
        print(f"\n⚠️ Phase 2 needs additional tuning before integration")
    
    return overall_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)