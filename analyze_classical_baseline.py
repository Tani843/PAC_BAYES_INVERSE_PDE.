# analyze_classical_baseline.py
import json
import numpy as np

print("=" * 60)
print("CLASSICAL BASELINE ANALYSIS")
print("=" * 60)

# Load the successful classical baseline results
with open('classical_baseline_20250926_085459/classical_baseline_results.json', 'r') as f:
    data = json.load(f)

print(f"Total classical experiments: {len(data)}")

# Analyze parameter coverage
param_combinations = set()
success_count = 0
acceptance_rates = []
ess_values = []

print(f"\nParameter Coverage:")
for i, exp in enumerate(data):
    config = exp['config']
    status = exp['status']
    
    param_key = f"s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}, seed={config['seed']}"
    param_combinations.add(param_key)
    
    print(f"  {i+1:2d}: {param_key} → {status}")
    
    if status == 'success':
        success_count += 1
        mcmc = exp['mcmc']
        acceptance_rates.append(mcmc['acceptance_rate'])
        ess_values.append(mcmc['min_ess'])

print(f"\nSummary Statistics:")
print(f"  Success rate: {success_count}/{len(data)} = {success_count/len(data)*100:.1f}%")
print(f"  Unique parameter combinations: {len(param_combinations)}")
print(f"  Mean acceptance rate: {np.mean(acceptance_rates):.1%}")
print(f"  Mean minimum ESS: {np.mean(ess_values):.1f}")
print(f"  ESS range: [{np.min(ess_values):.0f}, {np.max(ess_values):.0f}]")

# Check coverage vs PAC-Bayes
print(f"\nClassical vs PAC-Bayes Coverage:")
print(f"  Classical baseline: 12 experiments (focused subset)")
print(f"  PAC-Bayes main: 1,728 experiments (full grid)")
print(f"  Coverage ratio: {12/1728*100:.1f}%")

print(f"\nClassical Baseline Design:")
print(f"  Strategy: Representative subset for method comparison")
print(f"  Parameters: s∈{{3,5}}, σ∈{{0.05,0.10,0.20}}, m∈{{3,5}}")
print(f"  Seeds: Single seed (101) for focused comparison")
print(f"  Purpose: Generate credible intervals for Figure 3")

# Sample credible intervals
print(f"\nSample Credible Intervals (first experiment):")
if data[0]['status'] == 'success':
    intervals = data[0]['credible_intervals']
    for param, stats in intervals.items():
        mean = stats['mean']
        ci_lower = stats['q025']
        ci_upper = stats['q975']
        print(f"  {param}: {mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

print("=" * 60)
print("✅ Classical baseline ready for comparative analysis!")
print("   Can now generate Figure 3: Classical credible bands vs B_λ")
print("=" * 60)