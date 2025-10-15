#!/usr/bin/env python3
"""
Comprehensive verification of all 1,728 PAC-Bayes inverse PDE experiments
"""

import json
from pathlib import Path
import numpy as np
from collections import Counter

def verify_complete_experiments():
    """Comprehensive verification of all 1,728 experiments"""
    
    print("=" * 60)
    print("VERIFYING COMPLETE PAC-BAYES EXPERIMENT GRID")
    print("=" * 60)
    
    all_experiments = {}
    
    # 1. Load missing experiments (0-81)
    missing_dir = sorted(Path('.').glob('missing_experiments_*'))[-1]
    missing_files = list(missing_dir.glob('exp_*.json'))
    print(f"\n1. Missing experiments directory: {missing_dir}")
    print(f"   Files found: {len(missing_files)}")
    
    for exp_file in missing_files:
        with open(exp_file) as f:
            exp_num = int(exp_file.stem.split('_')[1])
            all_experiments[exp_num] = json.load(f)
    
    # 2. Load main experiments (82-1727)
    main_dir = Path('results_phase2_final_20250921_151355')
    main_files = list(main_dir.glob('exp_*.json'))
    print(f"\n2. Main experiments directory: {main_dir}")
    print(f"   Files found: {len(main_files)}")
    
    for exp_file in main_files:
        with open(exp_file) as f:
            exp_num = int(exp_file.stem.split('_')[1])
            all_experiments[exp_num] = json.load(f)
    
    # 3. Verify continuity
    print("\n3. CONTINUITY CHECK:")
    exp_numbers = sorted(all_experiments.keys())
    expected_range = list(range(0, 1728))
    missing_nums = set(expected_range) - set(exp_numbers)
    
    if missing_nums:
        print(f"   ❌ MISSING experiments: {sorted(missing_nums)}")
    else:
        print(f"   ✅ Perfect continuity: 0 to {max(exp_numbers)}")
    
    # 4. Verify parameter coverage
    print("\n4. PARAMETER COVERAGE:")
    params = {
        's': [], 'sigma': [], 'lambda': [], 'T': [],
        'm': [], 'n_x': [], 'n_t': [], 'seed': []
    }
    
    for exp in all_experiments.values():
        config = exp['config']
        params['s'].append(config['s'])
        params['sigma'].append(config['sigma'])
        params['lambda'].append(config['lambda'])
        params['T'].append(config['T'])
        params['m'].append(config['m'])
        params['n_x'].append(config.get('n_x', 50))
        params['n_t'].append(config.get('n_t', 50))
        params['seed'].append(config['seed'])
    
    for param, values in params.items():
        unique = sorted(set(values))
        counts = Counter(values)
        print(f"   {param}: {unique}")
        if len(unique) <= 10:  # Only show distribution for manageable number of values
            print(f"        Distribution: {dict(counts)}")
    
    # 5. Verify expected grid size
    print("\n5. GRID SIZE VERIFICATION:")
    s_vals = len(set(params['s']))
    sigma_vals = len(set(params['sigma']))
    lambda_vals = len(set(params['lambda']))
    T_vals = len(set(params['T']))
    m_vals = len(set(params['m']))
    seed_vals = len(set(params['seed']))
    
    expected_combinations = s_vals * sigma_vals * lambda_vals * T_vals * m_vals * seed_vals
    print(f"   Parameter dimensions: {s_vals}×{sigma_vals}×{lambda_vals}×{T_vals}×{m_vals}×{seed_vals}")
    print(f"   Expected: {expected_combinations} experiments")
    print(f"   Actual: {len(all_experiments)} experiments")
    print(f"   Status: {'✅ COMPLETE' if len(all_experiments) == 1728 else '❌ INCOMPLETE'}")
    
    # 6. Check success rates
    print("\n6. SUCCESS RATE ANALYSIS:")
    statuses = Counter(exp.get('status', 'unknown') for exp in all_experiments.values())
    total = len(all_experiments)
    
    for status, count in statuses.items():
        percentage = (count / total) * 100
        symbol = "✅" if status == 'success' else "⚠️"
        print(f"   {symbol} {status}: {count}/{total} ({percentage:.1f}%)")
    
    # 7. Runtime analysis
    print("\n7. RUNTIME ANALYSIS:")
    runtimes = []
    for exp in all_experiments.values():
        if exp.get('status') == 'success' and 'performance' in exp:
            runtime = exp['performance'].get('runtime', 0)
            if runtime > 0:
                runtimes.append(runtime)
    
    if runtimes:
        print(f"   Successful experiments with runtime data: {len(runtimes)}")
        print(f"   Average runtime: {np.mean(runtimes):.1f}s")
        print(f"   Runtime range: {np.min(runtimes):.1f}s - {np.max(runtimes):.1f}s")
        print(f"   Total compute time: {sum(runtimes)/3600:.1f} hours")
    
    # 8. Final validation
    print("\n" + "=" * 60)
    if len(all_experiments) == 1728 and not missing_nums:
        print("🎉 VALIDATION PASSED: Complete 1,728 experiment grid!")
        print("🎉 Ready for analysis and publication!")
        
        # Save the complete dataset
        print("\n📁 Saving complete verified dataset...")
        sorted_results = [all_experiments[i] for i in sorted(all_experiments.keys())]
        
        # Create comprehensive output file
        output_file = 'PAC_BAYES_COMPLETE_VERIFIED_1728.json'
        with open(output_file, 'w') as f:
            json.dump(sorted_results, f, indent=2)
        
        file_size = Path(output_file).stat().st_size / (1024*1024)  # MB
        print(f"✅ Saved to: {output_file}")
        print(f"✅ File size: {file_size:.1f} MB")
        print(f"✅ Contains {len(sorted_results)} experiments in order (exp_0000 to exp_1727)")
        
    else:
        print("❌ VALIDATION FAILED: Grid incomplete")
        if missing_nums:
            print(f"   Missing experiments: {sorted(list(missing_nums))}")
        if len(all_experiments) != 1728:
            print(f"   Expected 1728, found {len(all_experiments)}")
    
    print("=" * 60)
    return len(all_experiments) == 1728 and not missing_nums

if __name__ == '__main__':
    is_complete = verify_complete_experiments()
    if is_complete:
        print("\n🏆 MISSION ACCOMPLISHED: Complete PAC-Bayes dataset ready!")
    else:
        print("\n❌ Validation failed - please check for missing experiments")