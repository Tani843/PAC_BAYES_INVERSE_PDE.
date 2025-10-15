#!/usr/bin/env python3
"""
Spot-check specific experiments to verify complete parameter coverage
"""

import json

def main():
    print("=" * 80)
    print("VERIFYING COMPLETE PARAMETER COVERAGE")
    print("=" * 80)
    
    # Load results
    with open('results_full_section_a/section_a_complete_20250917_142211.json', 'r') as f:
        data = json.load(f)

    # Check for all expected parameter combinations
    expected_params = set()
    for s in [3, 5]:
        for placement in ['fixed', 'shifted']:
            for sigma in [0.05, 0.1, 0.2]:
                for nx in [50, 100]:
                    for T in [0.3, 0.5]:
                        for lam in [0.5, 1.0, 2.0]:
                            for m in [3, 5]:
                                for nt in [50, 100]:
                                    for seed in [101, 202, 303]:
                                        expected_params.add((s, placement, sigma, nx, T, lam, m, nt, seed))

    actual_params = set()
    for exp in data:
        c = exp['config']
        actual_params.add((c['s'], c['placement_type'], c['sigma'], c['n_x'], c['T'], c['lambda'], c['m'], c['n_t'], c['seed']))

    print(f"Expected parameters: {len(expected_params)}")
    print(f"Actual parameters: {len(actual_params)}")
    print(f"Perfect match: {expected_params == actual_params}")
    
    # Check for missing parameters
    missing = expected_params - actual_params
    if missing:
        print(f"\n❌ Missing {len(missing)} parameter combinations:")
        for params in sorted(missing)[:10]:  # Show first 10
            print(f"   s={params[0]}, placement={params[1]}, σ={params[2]}, n_x={params[3]}, T={params[4]}, λ={params[5]}, m={params[6]}, n_t={params[7]}, seed={params[8]}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")
    
    # Check for extra parameters  
    extra = actual_params - expected_params
    if extra:
        print(f"\n⚠️ Extra {len(extra)} parameter combinations:")
        for params in sorted(extra)[:10]:  # Show first 10
            print(f"   s={params[0]}, placement={params[1]}, σ={params[2]}, n_x={params[3]}, T={params[4]}, λ={params[5]}, m={params[6]}, n_t={params[7]}, seed={params[8]}")
        if len(extra) > 10:
            print(f"   ... and {len(extra) - 10} more")
    
    # Verify grid structure calculation
    expected_count = 2 * 2 * 3 * 2 * 2 * 3 * 1 * 2 * 2 * 3  # s × placement × σ × n_x × T × λ × c × m × n_t × seed
    print(f"\nGrid structure verification:")
    print(f"Expected count (2×2×3×2×2×3×1×2×2×3): {expected_count}")
    print(f"Actual count: {len(actual_params)}")
    print(f"Match: {expected_count == len(actual_params)}")
    
    if expected_params == actual_params:
        print(f"\n✅ PERFECT COVERAGE: All {len(expected_params)} parameter combinations present!")
        print("✅ Section A experimental grid is complete and correct.")
        return True
    else:
        print(f"\n❌ INCOMPLETE COVERAGE: Parameter mismatch detected.")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)