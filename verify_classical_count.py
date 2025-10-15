# verify_classical_count.py
import json
from pathlib import Path

# Find the classical baseline results directory
classical_dirs = list(Path('.').glob('classical_baseline_*'))
print(f"Found {len(classical_dirs)} classical baseline directories:")
for dir in classical_dirs:
    print(f"  - {dir}")

# Check the most recent one
if classical_dirs:
    latest_dir = sorted(classical_dirs)[-1]
    
    # Check for the final results file
    results_files = list(latest_dir.glob('*.json'))
    print(f"\nFiles in {latest_dir}:")
    for f in results_files:
        print(f"  - {f.name}")
    
    # Load and count experiments
    for f in results_files:
        if 'classical_baseline' in f.name or 'results' in f.name:
            with open(f, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    print(f"\n{f.name}: {len(data)} experiments")
                    
                    # Check parameter coverage
                    params = {}
                    for exp in data:
                        if 'config' in exp:
                            config = exp['config']
                            key = f"s={config.get('s')}, σ={config.get('sigma')}, m={config.get('m')}"
                            params[key] = params.get(key, 0) + 1
                    
                    print("Parameter combinations:")
                    for k, v in sorted(params.items()):
                        print(f"  {k}: {v} experiments")

# Expected: 72 experiments
# s∈{3,5} × σ∈{0.05,0.10,0.20} × m∈{3,5} × seeds∈{101,202,303}
# = 2 × 3 × 2 × 3 = 36 parameter combos × 3 seeds = 36 (not 72?)
# Actually: 2 × 3 × 2 × 3 = 36 unique configs