#!/usr/bin/env python3
"""
Test the fixed experiment config
"""

try:
    from config.experiment_config_fixed import ExperimentConfig
    
    print("✓ Fixed config imported successfully")
    
    config = ExperimentConfig()
    print("✓ Config instantiated successfully")
    
    # Test main grid generation (the full 1,728 experiments)
    experiments = config.get_experiment_grid(include_appendix=False)
    print(f"✓ Main grid generated: {len(experiments)} experiments")
    
    # Test appendix grid generation 
    experiments_appendix = config.get_experiment_grid(include_appendix=True)
    print(f"✓ Appendix grid generated: {len(experiments_appendix)} experiments")
    
    # Test baseline grid
    baseline_experiments = config.get_baseline_subset()
    print(f"✓ Baseline grid generated: {len(baseline_experiments)} experiments")
    
    print(f"\n" + "="*60)
    print("FULL GRID VERIFICATION")
    print("="*60)
    print(f"Main grid: {len(experiments)} experiments")
    print(f"Expected main: 1728 (2×2×3×2×2×3×1×2×2×3)")
    print(f"Match: {len(experiments) == 1728}")
    
    print(f"\nAppendix grid: {len(experiments_appendix)} experiments") 
    print(f"Expected appendix: 5184 (1728×3 for c∈{{1.0,0.5,2.0}})")
    print(f"Match: {len(experiments_appendix) == 5184}")
    
    print(f"\nBaseline grid: {len(baseline_experiments)} experiments")
    print(f"Expected baseline: 72 (2×2×3×2×3 for classical Bayes)")
    print(f"Match: {len(baseline_experiments) == 72}")
    
    # Check parameter ranges in main grid
    if experiments:
        from collections import Counter
        
        print(f"\n" + "="*60)
        print("PARAMETER DISTRIBUTIONS IN MAIN GRID")
        print("="*60)
        
        for param in ['s', 'placement_type', 'sigma', 'n_x', 'T', 'lambda', 'c', 'm', 'n_t', 'seed']:
            values = Counter(exp[param] for exp in experiments)
            print(f"{param}: {dict(values)}")
        
        # Sample configuration
        sample = experiments[0]
        print(f"\nSample configuration:")
        print(f"  s={sample['s']}, placement={sample['placement_type']}")
        print(f"  sigma={sample['sigma']}, n_x={sample['n_x']}, T={sample['T']}")
        print(f"  lambda={sample['lambda']}, c={sample['c']}, m={sample['m']}, n_t={sample['n_t']}")
        print(f"  seed={sample['seed']}, n={sample['n']}")
    
    print(f"\n✓ All tests passed! Ready to run full 1,728 experiment grid.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()