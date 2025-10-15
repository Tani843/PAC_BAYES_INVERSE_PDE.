# debug_grid.py
from config.experiment_config import ExperimentConfig
import itertools

config = ExperimentConfig()

# Check what parameters are actually being varied
experiments = config.get_experiment_grid(include_appendix=False)

# Get unique parameter combinations
param_sets = set()
for exp in experiments:
    params = (
        exp['s'],
        exp['placement_type'],
        exp['sigma'],
        exp['n_x'],
        exp['T'],
        exp['lambda'],
        exp['m'],
        exp['n_t'],
        exp['seed']
    )
    param_sets.add(params)

print(f"Total experiments: {len(experiments)}")
print(f"Unique parameter combinations: {len(param_sets)}")

# Count variations of each parameter
from collections import Counter
for param in ['s', 'sigma', 'n_x', 'T', 'lambda', 'm', 'n_t', 'seed']:
    values = Counter(exp[param] for exp in experiments)
    print(f"{param}: {dict(values)}")