#!/usr/bin/env python3
"""
Test if we can import and use the experiment config
"""

try:
    from config.experiment_config import ExperimentConfig
    
    print("✓ Config imported successfully")
    
    config = ExperimentConfig()
    print("✓ Config instantiated successfully")
    
    # Test main grid generation
    experiments = config.get_experiment_grid(include_appendix=False)
    print(f"✓ Main grid generated: {len(experiments)} experiments")
    
    # Test appendix grid generation
    experiments_appendix = config.get_experiment_grid(include_appendix=True)
    print(f"✓ Appendix grid generated: {len(experiments_appendix)} experiments")
    
    print(f"\nGrid verification:")
    print(f"  Main grid: {len(experiments)} experiments")
    print(f"  Appendix grid: {len(experiments_appendix)} experiments")
    print(f"  Expected main: 1728 (2×2×3×2×2×3×1×2×2×3)")
    print(f"  Expected appendix: 5184 (1728×3 for c∈{1.0,0.5,2.0})")
    
    # Check a sample configuration
    if experiments:
        sample = experiments[0]
        print(f"\nSample configuration keys: {list(sample.keys())}")
        print(f"Sample s={sample['s']}, sigma={sample['sigma']}, n_t={sample['n_t']}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Config file has encoding or syntax issues")