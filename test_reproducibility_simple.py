#!/usr/bin/env python3
"""
Simple reproducibility test for Section J compliance verification
"""

import sys
import numpy as np

# Add project to path
sys.path.append('.')

from src.utils.reproducibility import RandomStateManager, ExperimentTracker

def test_fixed_seeds():
    """Test that fixed seeds {101, 202, 303} produce deterministic results."""
    print("Testing fixed seeds...")
    
    seeds = [101, 202, 303]
    results = []
    
    for seed in seeds:
        rng = RandomStateManager(seed)
        data = rng.get_data_rng().randn(100)
        results.append(data)
    
    # Verify each seed produces different results
    assert not np.allclose(results[0], results[1])
    assert not np.allclose(results[1], results[2])
    print("✓ Different seeds produce different results")
    
    # But same seed produces identical results
    rng1 = RandomStateManager(101)
    rng2 = RandomStateManager(101)
    data1 = rng1.get_data_rng().randn(100)
    data2 = rng2.get_data_rng().randn(100)
    
    assert np.allclose(data1, data2)
    print("✓ Same seed produces identical results")
    
    return True

def test_separate_rng_streams():
    """Test that RNG streams are properly separated."""
    print("\nTesting separate RNG streams...")
    
    rng_manager = RandomStateManager(base_seed=101)
    
    # Get initial states
    data_state1 = rng_manager.get_data_rng().get_state()
    prior_state1 = rng_manager.get_prior_rng().get_state()
    mcmc_state1 = rng_manager.get_mcmc_rng().get_state()
    
    # Use data RNG
    data_samples = rng_manager.get_data_rng().randn(10)
    
    # Check that only data RNG state changed
    data_state2 = rng_manager.get_data_rng().get_state()
    prior_state2 = rng_manager.get_prior_rng().get_state()
    mcmc_state2 = rng_manager.get_mcmc_rng().get_state()
    
    # Data state should have changed
    assert not np.array_equal(data_state1[1], data_state2[1])
    print("✓ Data RNG state changed after use")
    
    # Other states should be unchanged
    assert np.array_equal(prior_state1[1], prior_state2[1])
    assert np.array_equal(mcmc_state1[1], mcmc_state2[1])
    print("✓ Other RNG states unchanged")
    
    # Verify usage log (called once for data samples + calls for state checks)
    assert rng_manager.usage_log['data'] >= 1  # At least called once
    print("✓ Usage log tracking works")
    
    return True

def test_experiment_tracker():
    """Test basic experiment tracker functionality."""
    print("\nTesting experiment tracker...")
    
    config = {
        'seed': 101,
        'test': True
    }
    
    # Create tracker
    tracker = ExperimentTracker(config)
    
    # Test RNG streams
    data_rng = tracker.get_rng('data')
    prior_rng = tracker.get_rng('prior')
    mcmc_rng = tracker.get_rng('mcmc')
    
    # Generate some random numbers
    data_vals = data_rng.randn(5)
    prior_vals = prior_rng.randn(5)
    mcmc_vals = mcmc_rng.randn(5)
    
    # Verify different streams produce different values
    assert not np.allclose(data_vals, prior_vals)
    assert not np.allclose(prior_vals, mcmc_vals)
    print("✓ Different RNG streams produce different values")
    
    # Test hash computation
    hash1 = tracker.compute_hash(np.array([1, 2, 3]))
    hash2 = tracker.compute_hash(np.array([1, 2, 3]))
    hash3 = tracker.compute_hash(np.array([1, 2, 4]))
    
    assert hash1 == hash2  # Same data -> same hash
    assert hash1 != hash3  # Different data -> different hash
    print("✓ Hash computation works correctly")
    
    return True

def main():
    """Run all reproducibility tests"""
    print("="*60)
    print("REPRODUCIBILITY VERIFICATION TEST")
    print("="*60)
    
    tests = [
        test_fixed_seeds,
        test_separate_rng_streams,
        test_experiment_tracker
    ]
    
    passed = 0
    try:
        for test in tests:
            if test():
                passed += 1
        
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("✓ All reproducibility tests passed!")
            print("✓ Section J compliance verified!")
            return True
        else:
            print("✗ Some tests failed")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)