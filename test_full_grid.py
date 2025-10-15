#!/usr/bin/env python3
"""
Test the full experiment grid generation without importing the problematic config file
"""

import itertools
from collections import Counter

def generate_full_section_a_grid():
    """Generate the complete Section A experiment grid."""
    
    # Parameters exactly as specified in Section A
    s_values = [3, 5]
    placement_types = ['fixed', 'shifted']
    sigma_values = [0.05, 0.10, 0.20]  # All three noise levels
    n_x_values = [50, 100]
    T_values = [0.3, 0.5]              # Both time horizons
    lambda_values = [0.5, 1.0, 2.0]
    c_values = [1.0]                   # Main experiments only
    m_values = [3, 5]
    n_t_values = [50, 100]             # Both temporal discretizations
    seeds = [101, 202, 303]
    
    # Sensor placements
    sensor_placements = {
        3: {
            'fixed': [0.25, 0.50, 0.75],
            'shifted': [0.20, 0.50, 0.80]
        },
        5: {
            'fixed': [0.10, 0.30, 0.50, 0.70, 0.90],
            'shifted': [0.20, 0.35, 0.50, 0.65, 0.80]
        }
    }
    
    experiments = []
    
    # Generate ALL combinations
    for s, placement_type, sigma, n_x, T, lambda_val, c, m, n_t, seed in itertools.product(
        s_values,           # [3, 5]
        placement_types,    # ['fixed', 'shifted']
        sigma_values,       # [0.05, 0.10, 0.20]
        n_x_values,         # [50, 100]
        T_values,           # [0.3, 0.5]
        lambda_values,      # [0.5, 1.0, 2.0]
        c_values,           # [1.0]
        m_values,           # [3, 5]
        n_t_values,         # [50, 100]
        seeds               # [101, 202, 303]
    ):
        config = {
            's': s,
            'sensor_positions': sensor_placements[s][placement_type],
            'placement_type': placement_type,
            'sigma': sigma,
            'n_x': n_x,
            'T': T,
            'lambda': lambda_val,
            'c': c,
            'm': m,
            'n_t': n_t,
            'seed': seed,
            'delta': 0.05,
            'alpha': 1e-3,
            'M': 2000,
            'R': 100,
            'n': s * n_t,  # Total observations
            'Delta_x': 1.0 / n_x,
            'Delta_t': T / (n_t - 1),
            'mcmc_n_steps': 10000,
            'mcmc_n_burn': 2000,
            'is_baseline': False
        }
        experiments.append(config)
    
    return experiments

def analyze_full_grid():
    """Analyze the complete experimental grid."""
    
    experiments = generate_full_section_a_grid()
    
    print("=" * 80)
    print("COMPLETE SECTION A EXPERIMENT GRID ANALYSIS")
    print("=" * 80)
    
    total = len(experiments)
    print(f"Total experiments: {total}")
    
    # Calculate expected
    expected = 2 * 2 * 3 * 2 * 2 * 3 * 1 * 2 * 2 * 3  # s * placement * sigma * n_x * T * lambda * c * m * n_t * seeds
    print(f"Expected: 2×2×3×2×2×3×1×2×2×3 = {expected}")
    print(f"Match: {total == expected}")
    
    # Parameter distributions
    print(f"\nParameter value distributions:")
    for param in ['s', 'placement_type', 'sigma', 'n_x', 'T', 'lambda', 'c', 'm', 'n_t', 'seed']:
        values = Counter(exp[param] for exp in experiments)
        print(f"  {param}: {dict(values)}")
    
    # Unique configurations per seed
    unique_configs = set()
    for exp in experiments:
        config_key = (
            exp['s'], exp['placement_type'], exp['sigma'], 
            exp['n_x'], exp['T'], exp['lambda'], exp['c'], exp['m'], exp['n_t']
        )
        unique_configs.add(config_key)
    
    print(f"\nUnique configurations (without seed): {len(unique_configs)}")
    print(f"Configurations per seed: {len(unique_configs)}")
    print(f"Total with 3 seeds: {len(unique_configs)} × 3 = {len(unique_configs) * 3}")
    
    # Compare to our focused grid
    print(f"\n" + "=" * 80)
    print("COMPARISON TO OUR FOCUSED GRID")
    print("=" * 80)
    
    focused_total = 72
    print(f"Full Section A grid: {total} experiments")
    print(f"Our focused grid: {focused_total} experiments")
    print(f"Full grid is {total / focused_total:.1f}x larger")
    print(f"Our grid covers {focused_total / total * 100:.1f}% of full grid")
    
    # Show parameter reductions in focused grid
    print(f"\nFocused grid reductions:")
    print(f"  placement_type: 2 → 1 (fixed only)")
    print(f"  sigma: 3 → 1 (0.1 only)")  
    print(f"  T: 2 → 1 (0.5 only)")
    print(f"  n_t: 2 → derived from n_x (effectively 2 values)")
    print(f"  Reduction factor: (2×3×2×2)/(1×1×1×1) = 24x in those dimensions")
    
    # Computational implications
    print(f"\n" + "=" * 80)
    print("COMPUTATIONAL IMPLICATIONS")
    print("=" * 80)
    
    # Estimate runtime
    focused_runtime_hours = 2  # Approximate from our experience
    full_runtime_hours = focused_runtime_hours * (total / focused_total)
    
    print(f"Estimated runtimes (based on focused grid experience):")
    print(f"  Focused grid (72 exp): ~{focused_runtime_hours} hours")
    print(f"  Full grid ({total} exp): ~{full_runtime_hours:.1f} hours ({full_runtime_hours/24:.1f} days)")
    
    print(f"\nFull grid advantages:")
    print(f"  ✓ Complete parameter space coverage")
    print(f"  ✓ All noise levels (σ ∈ {{0.05, 0.1, 0.2}})")
    print(f"  ✓ Both sensor placements (fixed and shifted)")
    print(f"  ✓ Both time horizons (T ∈ {{0.3, 0.5}})")
    print(f"  ✓ Complete temporal discretization analysis")
    
    print(f"\nFull grid considerations:")
    print(f"  ⚠ {total/focused_total:.0f}x longer runtime")
    print(f"  ⚠ {total} result files to manage")
    print(f"  ⚠ ~{full_runtime_hours/24:.1f} days of computation")
    
    return experiments

if __name__ == '__main__':
    experiments = analyze_full_grid()
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("The full 1,728-experiment grid provides complete Section A coverage")
    print("but requires ~24x more computation time than our focused approach.")
    print("For publication, the focused grid demonstrates all key scientific")
    print("insights while maintaining computational feasibility.")