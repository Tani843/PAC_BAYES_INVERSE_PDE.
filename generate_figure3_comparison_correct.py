import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure output dir exists
os.makedirs('paper_figures', exist_ok=True)

# Load both datasets
with open('classical_baseline_complete_72_20250926_144444/classical_baseline_72_complete.json', 'r') as f:
    classical = json.load(f)

with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    pacbayes_all = json.load(f)

# Filter PAC-Bayes to matching configs (n_x=100, T=0.5, λ=1.0, fixed placement)
pacbayes = [exp for exp in pacbayes_all
            if exp.get('status') == 'success'
            and exp['config']['n_x'] == 100
            and exp['config']['T'] == 0.5
            and exp['config']['lambda'] == 1.0
            and exp['config'].get('placement_type') == 'fixed']

# Create figure (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Figure 3: Classical Credible Intervals vs PAC-Bayes Certificates (λ=1.0)', fontsize=14)

sigma_values = [0.05, 0.10, 0.20]
s_values = [3, 5]

# Default for caption in case no seed is found
seed_used_for_caption = '—'

for i, s in enumerate(s_values):
    for j, sigma in enumerate(sigma_values):
        ax = axes[i, j]
        ax2 = None  # (1) reset secondary axis per subplot

        # Select data for this configuration
        classical_subset = [c for c in classical
                            if c['config']['s'] == s
                            and c['config']['sigma'] == sigma
                            and c['config']['m'] == 3
                            and c['status'] == 'success']

        pacbayes_subset = [p for p in pacbayes
                           if p['config']['s'] == s
                           and p['config']['sigma'] == sigma
                           and p['config']['m'] == 3]

        if classical_subset and pacbayes_subset:
            # Use first (or you can aggregate across seeds if you wish)
            classical_exp = classical_subset[0]

            # (2) track the actual seed used for caption
            seed_used_for_caption = classical_exp['config'].get('seed', seed_used_for_caption)

            # Plot Classical 95% credible intervals on primary axis
            for k in range(3):  # m=3 parameters
                # FIX: use correct keys
                ci = classical_exp['credible_intervals'][f'kappa_{k}']
                mean = ci['mean']
                lower = ci['q025']
                upper = ci['q975']
                ax.errorbar(
                    k, mean,
                    yerr=[[mean - lower], [upper - mean]],
                    fmt='o', color='blue', capsize=5, capthick=2,
                    label='Classical 95% CI' if k == 0 else ''
                )

            # Secondary axis for PAC-Bayes risk (L̂ and B_λ)
            ax2 = ax.twinx()

            # Use first PAC-Bayes exp (or average)
            pb_exp = pacbayes_subset[0]
            cert = pb_exp.get('certificate', {})

            # Robust key access
            B_lambda = cert.get('B_lambda',
                        cert.get('B_lambda_val', cert.get('bound', None)))
            L_hat = cert.get('L_hat',
                      cert.get('L_hat_val', cert.get('empirical_risk', None)))

            if B_lambda is not None and L_hat is not None:
                # Optional: warn if values look out of [0,1]
                if not (0 <= L_hat <= 1 and 0 <= B_lambda <= 1):
                    print(f"Warning: Risk values out of [0,1] range for s={s}, σ={sigma}: "
                          f"L̂={L_hat}, B_λ={B_lambda}")

                ax2.axhline(y=L_hat, color='green', linestyle='--', linewidth=2,
                            alpha=0.8, label='L̂ (empirical)')
                # FIX: correct variable name
                ax2.axhline(y=B_lambda, color='red', linestyle='-', linewidth=2,
                            alpha=0.8, label='Bλ (certificate)')

                # (3) robust shading between L̂ and B_λ on secondary axis
                x0, x1 = ax.get_xlim()
                low, high = min(L_hat, B_lambda), max(L_hat, B_lambda)
                low, high = np.clip([low, high], 0.0, 1.0)
                ax2.fill_between([x0, x1], low, high, color='red', alpha=0.15, label='Certificate gap')
                ax2.set_ylim(0, 1.0)

                # Minor suggestion: right y-axis label (only on rightmost column to avoid clutter)
                if j == 2:
                    ax2.set_ylabel('Risk (L̂, B_λ)')
            else:
                # Optional warning to console (does not alter plot)
                print(f"Warning: Missing certificate values for s={s}, σ={sigma}. Keys: {list(cert.keys())}")

        # Axes cosmetics
        ax.set_xlabel('Parameter κ_j index (j=0,1,2)')
        ax.set_ylabel('Parameter value' if j == 0 else '')
        ax.set_title(f's={s}, σ={sigma:.2f}, m=3')
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 2.5)
        ax.grid(True, alpha=0.3)

        # (4) Legend handling (combine legends only on top-left subplot)
        if i == 0 and j == 0 and ax2 is not None:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines1 or lines2:
                ax.legend(lines1 + lines2, labels1 + labels2,
                          loc='upper left', fontsize=8, framealpha=0.9)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

        # (6) Annotate missing PAC-Bayes risk on any subplot where it wasn't drawn
        if ax2 is None:
            ax.text(0.5, 0.9, 'No PAC-Bayes risk available',
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=8, color='gray')

# (5) Layout padding to leave space for title & caption
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Caption with actual seed used (if found)
fig.text(
    0.5, 0.01,
    f'Note: Results shown for seed {seed_used_for_caption}. Classical: parameter credible intervals. PAC-Bayes: risk bounds.',
    ha='center', fontsize=9, style='italic'
)

# Save
plt.savefig('paper_figures/figure3_classical_vs_pacbayes_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper_figures/figure3_classical_vs_pacbayes_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure 3 generated with proper dual axes:")
print("- Left y-axis: Parameter values (κj) with Classical 95% CI")
print("- Right y-axis: Risk measures (L̂, Bλ) from PAC-Bayes")
print("- Legends combined on the top-left panel; subplots annotate if PAC-Bayes risk missing.")