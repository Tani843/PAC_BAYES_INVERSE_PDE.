# verify_appendix_grid.py
from config.experiment_config import ExperimentConfig

config = ExperimentConfig()
experiments_appendix = config.get_experiment_grid(include_appendix=True)

print(f"Appendix grid experiments: {len(experiments_appendix)}")
print(f"Expected: 1,728 × 3 = 5,184 (for c ∈ {{1.0, 0.5, 2.0}})")
print(f"Match: {len(experiments_appendix) == 5184}")

# Verify c parameter distribution
from collections import Counter
c_values = Counter(exp['c'] for exp in experiments_appendix)
print(f"c parameter distribution: {dict(c_values)}")

# Verify we have the right ratio
print(f"c=1.0: {c_values[1.0]} experiments")
print(f"c=0.5: {c_values[0.5]} experiments") 
print(f"c=2.0: {c_values[2.0]} experiments")
print(f"Each c value should have 1,728 experiments: {all(count == 1728 for count in c_values.values())}")