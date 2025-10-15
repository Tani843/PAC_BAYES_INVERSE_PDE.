#!/usr/bin/env python3
"""
Simple classical baseline runner using successful PAC-Bayes infrastructure
"""

import json
import numpy as np
from datetime import datetime

print("=" * 60)
print("CLASSICAL BASELINE CONFIGURATION")
print("=" * 60)

# Generate classical baseline configurations
# Same parameter space as PAC-Bayes but without lambda (temperature parameter)
baseline_configs = []

for s in [3, 5]:  # sensor counts
    for sigma in [0.05, 0.10, 0.20]:  # noise levels
        for m in [3, 5]:  # measurement counts
            for seed in [101, 202, 303]:  # random seeds
                config = {
                    's': s,
                    'sigma': sigma,
                    'n_x': 100,  # fixed spatial resolution
                    'n_t': 50,   # fixed temporal resolution
                    'T': 0.5,    # fixed final time
                    'm': m,
                    'seed': seed,
                    'placement_type': 'fixed',
                    'delta': 0.05,
                    'n': s * 50,  # total observations
                    'method': 'classical'  # distinguish from PAC-Bayes
                }
                baseline_configs.append(config)

print(f"Generated {len(baseline_configs)} classical baseline configurations")
print(f"Parameter space:")
print(f"  s (sensors): [3, 5]")
print(f"  σ (noise): [0.05, 0.10, 0.20]") 
print(f"  m (measurements): [3, 5]")
print(f"  seeds: [101, 202, 303]")
print(f"  Fixed: n_x=100, T=0.5, placement='fixed'")

# Save configuration for future use
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_file = f'classical_baseline_configs_{timestamp}.json'

with open(config_file, 'w') as f:
    json.dump(baseline_configs, f, indent=2)

print(f"\n✅ Classical baseline configurations saved to: {config_file}")

# Key differences from PAC-Bayes:
print(f"\nKey differences from PAC-Bayes experiments:")
print(f"  - NO temperature parameter λ (classical uses standard posterior)")
print(f"  - Uses π(θ|y) ∝ exp(-L(y,F(θ)))π(θ) instead of Q_λ")
print(f"  - Generates credible intervals instead of certificates")
print(f"  - Same MCMC, forward model, and data generation")

print(f"\nTo run experiments:")
print(f"  1. Implement classical_posterior.py (✅ done)")
print(f"  2. Adapt existing MCMC infrastructure")  
print(f"  3. Generate credible bands for comparison with B_λ")

print(f"\nClassical baseline ready for Figure 3 comparison!")
print("=" * 60)