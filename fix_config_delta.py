#!/usr/bin/env python3
"""
Check current experiment configuration and verify if Delta_x/Delta_t are included.
"""

from config.experiment_config import ExperimentConfig

# Check current config generation
config = ExperimentConfig()
experiments = config.get_experiment_grid()

# Verify Delta_x is included
test_exp = experiments[0]
print(f"Total experiments: {len(experiments)}")
print(f"Sample experiment keys: {list(test_exp.keys())}")
print(f"Delta_x present: {'Delta_x' in test_exp}")
print(f"Delta_t present: {'Delta_t' in test_exp}")

# Show sample config
print(f"\nSample config:")
for key, value in test_exp.items():
    print(f"  {key}: {value}")

if 'Delta_x' not in test_exp:
    print(f"\n❌ Delta_x missing from configuration!")
    print(f"   Need to add: 'Delta_x': 1.0 / n_x")
    
if 'Delta_t' not in test_exp:
    print(f"❌ Delta_t missing from configuration!")
    print(f"   Need to add: 'Delta_t': T / (n_t - 1)")