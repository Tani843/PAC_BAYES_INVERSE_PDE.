#!/usr/bin/env python3
"""
Test single appendix C experiment to verify functionality.
"""

from run_appendix_c_sensitivity import get_appendix_c_experiments, run_single_c_experiment
from pathlib import Path

def test_single_appendix_c():
    """Test running a single appendix C experiment."""
    
    # Get first experiment configuration
    experiments = get_appendix_c_experiments()
    config = experiments[0]  # First experiment: s=3, σ=0.05, m=3, c=0.5, seed=101
    
    print(f"Testing single experiment:")
    print(f"  s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}")
    print(f"  c={config['c']:.1f}, seed={config['seed']}")
    
    # Create test output directory
    output_dir = Path('test_appendix_c_output')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run single experiment
        result = run_single_c_experiment(config, output_dir)
        
        if 'error' in result:
            print(f"✗ Failed: {result['error']}")
            return False
        else:
            cert = result['certificate']
            print(f"✓ Success!")
            print(f"  L̂={cert['L_hat']:.4f}")
            print(f"  B_λ={cert['B_lambda']:.4f}")
            print(f"  Valid={cert['valid']}")
            print(f"  Runtime={result['performance']['runtime']:.1f}s")
            return True
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

if __name__ == '__main__':
    success = test_single_appendix_c()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")