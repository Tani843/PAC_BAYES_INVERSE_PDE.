# generate_figure3_final.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load both datasets
with open('classical_baseline_complete_72_20250926_144444/classical_baseline_72_complete.json', 'r') as f:
    classical = json.load(f)

with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    pacbayes = json.load(f)

print("=== FIGURE 3: CLASSICAL vs PAC-BAYES COMPARISON ===")
print(f"Classical baseline: {len(classical)} experiments")
print(f"PAC-Bayes dataset: {len(pacbayes)} experiments")

# Create output directory
output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)

# Helper function to match experiments by configuration
def find_matching_pacbayes(classical_config):
    """Find PAC-Bayes experiments that match classical configuration"""
    matches = []
    for pb_exp in pacbayes:
        if (pb_exp['config']['s'] == classical_config['s'] and
            pb_exp['config']['sigma'] == classical_config['sigma'] and
            pb_exp['config']['m'] == classical_config['m'] and
            pb_exp['config']['seed'] == classical_config['seed'] and
            pb_exp.get('status') == 'success'):
            matches.append(pb_exp)
    return matches

# Create Figure 3: Classical Credible Intervals vs PAC-Bayes Certificates
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Figure 3: Classical Credible Intervals vs PAC-Bayes Certificates', fontsize=16, fontweight='bold')

# Select representative experiments for comparison
selected_configs = [
    {'s': 3, 'sigma': 0.05, 'm': 3, 'seed': 101},
    {'s': 3, 'sigma': 0.10, 'm': 3, 'seed': 101}, 
    {'s': 3, 'sigma': 0.20, 'm': 3, 'seed': 101},
    {'s': 5, 'sigma': 0.05, 'm': 3, 'seed': 101},
    {'s': 5, 'sigma': 0.10, 'm': 3, 'seed': 101},
    {'s': 5, 'sigma': 0.20, 'm': 3, 'seed': 101}
]

for idx, config in enumerate(selected_configs):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Find classical experiment
    classical_exp = None
    for c_exp in classical:
        if (c_exp['config']['s'] == config['s'] and
            c_exp['config']['sigma'] == config['sigma'] and
            c_exp['config']['m'] == config['m'] and
            c_exp['config']['seed'] == config['seed']):
            classical_exp = c_exp
            break
    
    if classical_exp is None or classical_exp['status'] != 'success':
        ax.text(0.5, 0.5, 'No classical data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}")
        continue
        
    # Find matching PAC-Bayes experiments (all lambda values)
    pb_matches = find_matching_pacbayes(config)
    
    if not pb_matches:
        ax.text(0.5, 0.5, 'No PAC-Bayes data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}")
        continue
    
    # Plot comparison for each parameter
    m = config['m']
    x_positions = np.arange(m)
    
    # Classical credible intervals (95%)
    classical_intervals = classical_exp['credible_intervals']
    classical_means = []
    classical_lowers = []
    classical_uppers = []
    
    for j in range(m):
        param_key = f'kappa_{j}'
        if param_key in classical_intervals:
            stats = classical_intervals[param_key]
            classical_means.append(stats['mean'])
            classical_lowers.append(stats['q025'])
            classical_uppers.append(stats['q975'])
        else:
            classical_means.append(0)
            classical_lowers.append(0)
            classical_uppers.append(0)
    
    classical_means = np.array(classical_means)
    classical_lowers = np.array(classical_lowers)
    classical_uppers = np.array(classical_uppers)
    
    # Plot classical credible intervals as blue bands
    ax.fill_between(x_positions, classical_lowers, classical_uppers, 
                   alpha=0.3, color='blue', label='Classical 95% CI')
    ax.plot(x_positions, classical_means, 'b-o', linewidth=2, markersize=6, 
           label='Classical posterior mean')
    
    # PAC-Bayes certificates for different lambda values
    lambda_values = []
    pb_certificates = {}
    
    for pb_exp in pb_matches:
        if 'certificate' in pb_exp and pb_exp['certificate'] is not None:
            lam = pb_exp['config']['lambda']
            lambda_values.append(lam)
            
            cert = pb_exp['certificate']
            if 'B_lambda' in cert:
                pb_certificates[lam] = cert['B_lambda']
    
    # Sort by lambda
    lambda_values = sorted(set(lambda_values))
    
    # Plot PAC-Bayes certificates for selected lambda values
    colors = ['red', 'orange', 'green']
    lambda_subset = lambda_values[::max(1, len(lambda_values)//3)][:3]
    
    for i, lam in enumerate(lambda_subset):
        if lam in pb_certificates:
            B_lambda = pb_certificates[lam]
            if isinstance(B_lambda, list) and len(B_lambda) == m:
                ax.plot(x_positions, B_lambda, colors[i] + '--s', linewidth=2, 
                       markersize=4, label=f'PAC-Bayes B_λ (λ={lam:.3f})')
    
    # True parameter (if available)
    if 'true_kappa' in classical_exp.get('config', {}):
        true_kappa = classical_exp['config']['true_kappa']
        if isinstance(true_kappa, list) and len(true_kappa) == m:
            ax.plot(x_positions, true_kappa, 'k-', linewidth=3, 
                   label='True κ', alpha=0.7)
    
    ax.set_xlabel('Parameter index j')
    ax.set_ylabel('Parameter value')
    ax.set_title(f"s={config['s']}, σ={config['sigma']:.2f}, m={config['m']}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'κ_{j}' for j in range(m)])

# Add overall legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=12, 
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Save the figure
pdf_path = output_dir / 'figure3_classical_vs_pacbayes.pdf'
png_path = output_dir / 'figure3_classical_vs_pacbayes.png'

plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"✅ Figure 3 saved:")
print(f"   📄 PDF: {pdf_path}")
print(f"   🖼️  PNG: {png_path}")

# Generate summary statistics
print("\n=== COMPARISON SUMMARY ===")

# Count successful experiments
classical_success = sum(1 for exp in classical if exp['status'] == 'success')
pacbayes_success = sum(1 for exp in pacbayes if exp.get('status') == 'success')

print(f"Classical baseline: {classical_success}/{len(classical)} = {classical_success/len(classical)*100:.1f}% success")
print(f"PAC-Bayes experiments: {pacbayes_success}/{len(pacbayes)} = {pacbayes_success/len(pacbayes)*100:.1f}% success")

# Analyze classical interval widths vs PAC-Bayes certificate bounds
classical_widths = []
for exp in classical:
    if exp['status'] == 'success' and 'credible_intervals' in exp:
        for param, stats in exp['credible_intervals'].items():
            if 'q025' in stats and 'q975' in stats:
                width = stats['q975'] - stats['q025']
                classical_widths.append(width)

print(f"\nClassical credible interval analysis:")
print(f"  Mean width: {np.mean(classical_widths):.3f}")
print(f"  Median width: {np.median(classical_widths):.3f}")
print(f"  Width range: [{np.min(classical_widths):.3f}, {np.max(classical_widths):.3f}]")

# PAC-Bayes certificate analysis
pb_cert_sizes = []
pb_lambda_values = []

for exp in pacbayes:
    if (exp.get('status') == 'success' and 
        'certificate' in exp and 
        exp['certificate'] is not None and
        'B_lambda' in exp['certificate']):
        
        B_lambda = exp['certificate']['B_lambda']
        pb_lambda_values.append(exp['config']['lambda'])
        
        if isinstance(B_lambda, list):
            pb_cert_sizes.extend(B_lambda)

if pb_cert_sizes:
    print(f"\nPAC-Bayes certificate analysis:")
    print(f"  Mean B_λ component: {np.mean(pb_cert_sizes):.3f}")
    print(f"  Median B_λ component: {np.median(pb_cert_sizes):.3f}")
    print(f"  B_λ range: [{np.min(pb_cert_sizes):.3f}, {np.max(pb_cert_sizes):.3f}]")
    print(f"  Lambda range: [{np.min(pb_lambda_values):.6f}, {np.max(pb_lambda_values):.6f}]")

plt.show()
print("\n🎯 Figure 3 generation complete!")