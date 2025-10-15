# verify_figures.py
import json
import numpy as np
import pandas as pd

# Load your complete dataset
with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    data = json.load(f)

print("VERIFICATION OF FIGURES AGAINST EXPERIMENTAL DATA")
print("=" * 60)

# 1. Verify data source
print(f"\n1. DATA SOURCE CHECK:")
print(f"   Total experiments loaded: {len(data)}")
print(f"   Successful experiments: {sum(1 for exp in data if exp.get('status') == 'success')}")

# 2. Check if certificate values exist
has_certificate = sum(1 for exp in data if 'certificate' in exp and 'B_lambda' in exp['certificate'])
print(f"\n2. CERTIFICATE DATA CHECK:")
print(f"   Experiments with certificates: {has_certificate}")
print(f"   Experiments with B_lambda: {sum(1 for exp in data if exp.get('certificate', {}).get('B_lambda') is not None)}")
print(f"   Experiments with L_hat: {sum(1 for exp in data if exp.get('certificate', {}).get('L_hat') is not None)}")

# 3. Sample actual values from your data
print(f"\n3. SAMPLE VALUES FROM YOUR DATA:")
for i in range(min(5, len(data))):
    if data[i].get('status') == 'success':
        cert = data[i].get('certificate', {})
        config = data[i].get('config', {})
        print(f"   Exp {i}: σ={config.get('sigma')}, s={config.get('s')}, " +
              f"B_λ={cert.get('B_lambda', 'N/A')}, L̂={cert.get('L_hat', 'N/A')}")

# 4. Check parameter coverage
print(f"\n4. PARAMETER COVERAGE IN DATA:")
params = {'sigma': set(), 's': set(), 'lambda': set(), 'n_x': set()}
for exp in data:
    if exp.get('status') == 'success':
        config = exp.get('config', {})
        params['sigma'].add(config.get('sigma'))
        params['s'].add(config.get('s'))
        params['lambda'].add(config.get('lambda'))
        params['n_x'].add(config.get('n_x'))

for param, values in params.items():
    print(f"   {param}: {sorted(values)}")

# 5. Verify Figure 1 data points
print(f"\n5. FIGURE 1 DATA VERIFICATION:")
for sigma in [0.05, 0.10, 0.20]:
    for s in [3, 5]:
        count = sum(1 for exp in data 
                   if exp.get('status') == 'success' 
                   and exp.get('config', {}).get('sigma') == sigma 
                   and exp.get('config', {}).get('s') == s)
        print(f"   σ={sigma}, s={s}: {count} data points")

# 6. Check if values are realistic
print(f"\n6. VALUE RANGE CHECKS:")
b_lambda_values = [exp.get('certificate', {}).get('B_lambda', 0) 
                   for exp in data if exp.get('status') == 'success' and 'certificate' in exp]
l_hat_values = [exp.get('certificate', {}).get('L_hat', 0) 
                for exp in data if exp.get('status') == 'success' and 'certificate' in exp]

if b_lambda_values:
    print(f"   B_λ range: [{min(b_lambda_values):.4f}, {max(b_lambda_values):.4f}]")
    print(f"   L̂ range: [{min(l_hat_values):.4f}, {max(l_hat_values):.4f}]")
    
    # Key check: B_λ ≥ L̂ (certificate validity)
    valid_certs = sum(1 for b, l in zip(b_lambda_values, l_hat_values) if b >= l)
    print(f"   Valid certificates (B_λ ≥ L̂): {valid_certs}/{len(b_lambda_values)} = {valid_certs/len(b_lambda_values)*100:.1f}%")
else:
    print("   WARNING: No certificate values found!")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")