#!/usr/bin/env python3
"""
Simple setup verification test
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.append('.')

def test_imports():
    """Test that core modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.utils.reproducibility import RandomStateManager
        print("✓ RandomStateManager imported")
    except Exception as e:
        print(f"✗ RandomStateManager: {e}")
        return False
    
    try:
        from src.utils.logging import DiagnosticsLogger, PerformanceTracker
        print("✓ Logging utilities imported")
    except Exception as e:
        print(f"✗ Logging utilities: {e}")
        return False
    
    return True

def test_rng_manager():
    """Test RNG manager functionality"""
    print("\nTesting RNG Manager...")
    
    try:
        from src.utils.reproducibility import RandomStateManager
        
        # Test reproducibility with same seed
        rng1 = RandomStateManager(101)
        rng2 = RandomStateManager(101)
        
        # Generate samples
        data1 = rng1.get_data_rng().randn(10)
        data2 = rng2.get_data_rng().randn(10)
        
        if np.allclose(data1, data2):
            print("✓ RNG reproducibility verified")
            return True
        else:
            print("✗ RNG not reproducible")
            return False
            
    except Exception as e:
        print(f"✗ RNG test failed: {e}")
        return False

def test_logging():
    """Test logging functionality"""
    print("\nTesting Logging...")
    
    try:
        from src.utils.logging import DiagnosticsLogger, PerformanceTracker
        
        # Test diagnostics logger
        logger = DiagnosticsLogger('test_logs')
        
        # Test performance tracker
        perf = PerformanceTracker()
        perf.start_timer('test')
        import time
        time.sleep(0.01)
        perf.end_timer('test')
        
        print("✓ Logging functionality works")
        
        # Clean up
        import shutil
        if Path('test_logs').exists():
            shutil.rmtree('test_logs')
            
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("PAC-BAYES INVERSE PDE SETUP VERIFICATION")
    print("="*50)
    
    tests = [
        test_imports,
        test_rng_manager, 
        test_logging
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All tests passed! Setup verified successfully.")
        return True
    else:
        print("✗ Some tests failed. Check the output above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)