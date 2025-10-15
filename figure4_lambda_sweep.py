# figure4_lambda_sweep.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the complete dataset
with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df_list = []
for exp in data:
    if exp.get('status') == 'success':
        row = {
            's': exp['config']['s'],
            'sigma': exp['config']['sigma'],
            'lambda': exp['config']['lambda'],
            'n_x': exp['config']['n_x'],
            'L_hat': exp.get('certificate', {}).get('L_hat', np.nan),
            'B_lambda': exp.get('certificate', {}).get('B_lambda', np.nan),
            'KL': exp.get('certificate', {}).get('KL', np.nan),
            'eta_h': exp.get('certificate', {}).get('eta_h', np.nan),
        }
        df_list.append(row)

df = pd.DataFrame(df_list)

# Create Figure 4: Lambda sweep decomposition
fig, ax = plt.subplots(figsize=(10, 6))

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Plot decomposition for each λ
lambda_vals = sorted(df['lambda'].unique())
width = 0.25
x_pos = np.arange(len(lambda_vals))

# Calculate mean values for each lambda
l_hat_means = []
kl_means = []
eta_h_means = []

for lambda_val in lambda_vals:
    subset = df[df['lambda'] == lambda_val]
    l_hat_means.append(subset['L_hat'].mean())
    kl_means.append(subset['KL'].mean())
    eta_h_means.append(subset['eta_h'].mean())

# Create grouped bar chart
bars1 = ax.bar(x_pos - width, l_hat_means, width, label='L̂ (Empirical Risk)', 
               color='skyblue', alpha=0.8)
bars2 = ax.bar(x_pos, kl_means, width, label='KL Divergence', 
               color='lightcoral', alpha=0.8)
bars3 = ax.bar(x_pos + width, eta_h_means, width, label='η_h (Discretization)', 
               color='lightgreen', alpha=0.8)

# Add value labels on bars
for i, (l_hat, kl, eta) in enumerate(zip(l_hat_means, kl_means, eta_h_means)):
    ax.text(i - width, l_hat + max(l_hat_means) * 0.01, f'{l_hat:.0f}', 
            ha='center', va='bottom', fontweight='bold')
    ax.text(i, kl + max(l_hat_means) * 0.01, f'{kl:.4f}', 
            ha='center', va='bottom', fontweight='bold')
    ax.text(i + width, eta + max(l_hat_means) * 0.01, f'{eta:.4f}', 
            ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('λ (Temperature Parameter)')
ax.set_ylabel('Certificate Component Values')
ax.set_title('F4: Certificate Decomposition across λ\n' +
            'Takeaway: L̂ dominates certificate, KL≈0 shows excellent posterior concentration')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{lam:.1f}' for lam in lambda_vals])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_lambda_sweep.pdf', bbox_inches='tight')
plt.savefig('figure4_lambda_sweep.png', bbox_inches='tight')

print("✅ Figure 4: Lambda sweep decomposition saved")

# Additional analysis
print(f"\nLambda Analysis:")
for lambda_val in lambda_vals:
    subset = df[df['lambda'] == lambda_val]
    gap = (subset['B_lambda'] - subset['L_hat']).mean()
    print(f"λ={lambda_val}: Mean gap (B_λ - L̂) = {gap:.6f}")
    print(f"         L̂ = {subset['L_hat'].mean():.1f}")
    print(f"         KL = {subset['KL'].mean():.6f}")
    print(f"         η_h = {subset['eta_h'].mean():.6f}")