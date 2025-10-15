# verify_full_grid.py
from config.experiment_config import ExperimentConfig

config = ExperimentConfig()
experiments = config.get_experiment_grid(include_appendix=False)

print(f"Total experiments: {len(experiments)}")
print(f"Expected: 2×2×3×2×2×3×1×2×2×3 = 1,728")
print(f"Match: {len(experiments) == 1728}")

# Verify each parameter is fully varied
from collections import Counter
for param in ['s', 'placement_type', 'sigma', 'n_x', 'T', 'lambda', 'm', 'n_t', 'seed']:
    values = Counter(exp[param] for exp in experiments)
    print(f"{param}: {dict(values)}")