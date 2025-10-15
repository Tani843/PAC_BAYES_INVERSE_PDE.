# check_grid.py
from config.experiment_config import ExperimentConfig

config = ExperimentConfig()
experiments = config.get_experiment_grid(include_appendix=False)

print(f"Total experiments generated: {len(experiments)}")

# Count by seed
from collections import Counter
seed_counts = Counter(exp['seed'] for exp in experiments)
print(f"Experiments per seed: {dict(seed_counts)}")

# Check if n_t is being varied
nt_counts = Counter(exp['n_t'] for exp in experiments)
print(f"n_t values: {dict(nt_counts)}")

# Check unique configurations
unique_configs = set()
for exp in experiments:
    key = (exp['s'], exp['placement_type'], exp['sigma'], 
           exp['n_x'], exp['T'], exp['lambda'], exp['m'], exp['n_t'])
    unique_configs.add(key)
print(f"Unique configurations (without seed): {len(unique_configs)}")

# Detailed breakdown
print("\nConfiguration breakdown:")
print(f"s values: {sorted(set(exp['s'] for exp in experiments))}")
print(f"placement_type values: {sorted(set(exp['placement_type'] for exp in experiments))}")
print(f"sigma values: {sorted(set(exp['sigma'] for exp in experiments))}")
print(f"n_x values: {sorted(set(exp['n_x'] for exp in experiments))}")
print(f"T values: {sorted(set(exp['T'] for exp in experiments))}")
print(f"lambda values: {sorted(set(exp['lambda'] for exp in experiments))}")
print(f"m values: {sorted(set(exp['m'] for exp in experiments))}")
print(f"n_t values: {sorted(set(exp['n_t'] for exp in experiments))}")
print(f"seed values: {sorted(set(exp['seed'] for exp in experiments))}")

# Calculate expected total
s_count = len(set(exp['s'] for exp in experiments))
placement_count = len(set(exp['placement_type'] for exp in experiments))
sigma_count = len(set(exp['sigma'] for exp in experiments))
nx_count = len(set(exp['n_x'] for exp in experiments))
T_count = len(set(exp['T'] for exp in experiments))
lambda_count = len(set(exp['lambda'] for exp in experiments))
m_count = len(set(exp['m'] for exp in experiments))
nt_count = len(set(exp['n_t'] for exp in experiments))
seed_count = len(set(exp['seed'] for exp in experiments))

expected_total = s_count * placement_count * sigma_count * nx_count * T_count * lambda_count * m_count * nt_count * seed_count

print(f"\nExpected total: {s_count} × {placement_count} × {sigma_count} × {nx_count} × {T_count} × {lambda_count} × {m_count} × {nt_count} × {seed_count} = {expected_total}")
print(f"Actual total: {len(experiments)}")
print(f"Match: {len(experiments) == expected_total}")

# Check if this is the full Section A grid
print(f"\nSection A grid verification:")
print(f"This {'is' if not config.c_appendix else 'is not'} the main grid (appendix={False})")
print(f"c value: {experiments[0]['c'] if experiments else 'N/A'} (should be 1.0 for main)")