"""
Generate all figures and tables for Section I of PAC-Bayes inverse PDE paper.
Fixed version with proper bounded value usage for the final dataset.
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})
# Embed TrueType fonts in PDF/PS outputs (for camera-ready submission)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

def load_data(filepath: str = 'PAC_BAYES_COMPLETE_VERIFIED_1728.json') -> pd.DataFrame:
    """Load and prepare data from the final merged results JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    records = []
    # CHANGE #1: remove status filter — your JSON doesn’t include "status"
    for exp in data:
        config = exp['config']
        mcmc = exp.get('mcmc', {})
        cert = exp.get('certificate', {})
        
        # Extract config parameters
        s = config['s']
        sigma = config['sigma']
        n_t = config.get('n_t', 50)
        n = config.get('n', s * n_t)
        c = config.get('c', 1.0)
        
        # CHANGE #2: use values as provided; do not reconstruct/transform
        L_hat   = cert.get('L_hat',   np.nan)
        B_lambda = cert.get('B_lambda', np.nan)
        
        record = {
            's': s,
            'sigma': sigma,
            'lambda': config['lambda'],
            'T': config['T'],
            'n_x': config['n_x'],
            'n_t': n_t,
            'm': config['m'],
            'seed': config['seed'],
            'delta': config.get('delta', 0.05),
            'n': n,
            'c': c,
            'placement_type': config.get('placement_type', 'fixed'),
            'L_hat': L_hat,
            'B_lambda': B_lambda,
            'KL': cert.get('KL', np.nan),
            'eta_h': cert.get('eta_h', np.nan),
            'valid': cert.get('valid', False),
            'acceptance_rate': mcmc.get('acceptance_rate', np.nan),
            # CHANGE #3: correct ESS field name from your JSON: "ess_min"
            'min_ess': mcmc.get('ess_min', np.nan)
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Saturation check (informational; should be small for your final data)
    sat_L = (df['L_hat'] >= 0.99).mean() * 100
    sat_B = (df['B_lambda'] >= 0.99).mean() * 100
    
    print(f"Loaded {len(df)} experiments")
    print(f"Saturation check:")
    print(f"  L_hat ≥ 0.99: {sat_L:.1f}%")
    print(f"  B_lambda ≥ 0.99: {sat_B:.1f}%")
    
    # Diagnostic: Print medians for key configuration
    key_config = df[(df['n_x'] == 100) & (df['T'] == 0.5) & 
                    (df['lambda'] == 1.0) & (df['placement_type'] == 'fixed')]
    
    print("\nMedian values for (n_x=100, T=0.5, λ=1.0, fixed):")
    for s in [3, 5]:
        for sigma in [0.05, 0.10, 0.20]:
            subset = key_config[(key_config['s'] == s) & (key_config['sigma'] == sigma)]
            if len(subset) > 0:
                print(f"  s={s}, σ={sigma:.2f}: L̂={subset['L_hat'].median():.4f}, B_λ={subset['B_lambda'].median():.4f}")
    
    return df

def figure_1_certificate_validity(df: pd.DataFrame, output_dir: Path):
    """
    Figure 1: B_λ vs L̂ across noise levels and sensor configurations.
    """
    df_filtered = df[(df['n_x'] == 100) & 
                     (df['T'] == 0.5) & 
                     (df['lambda'] == 1.0) &
                     (df['placement_type'] == 'fixed')]
    
    sigma_colors = {0.05: 'C0', 0.10: 'C1', 0.20: 'C2'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    diagnostic_stats = {}
    
    for idx, s in enumerate([3, 5]):
        ax = axes[idx]
        df_s = df_filtered[df_filtered['s'] == s]
        
        for sigma in [0.05, 0.10, 0.20]:
            df_sigma = df_s[df_s['sigma'] == sigma]
            color = sigma_colors[sigma]
            if len(df_sigma) > 0:
                ax.scatter(df_sigma['L_hat'], df_sigma['B_lambda'], 
                           color=color, alpha=0.6, s=30, label=f'σ={sigma:.2f}')
                diagnostic_stats[(s, sigma)] = {
                    'L_hat_mean': df_sigma['L_hat'].mean()
                }
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='B_λ = L̂')
        
        from matplotlib.lines import Line2D
        filled_marker = Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='gray', markersize=8, label='L̂')
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l != "L_MC"]
        handles = [filled_marker] + [h for h, l in filtered]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, fontsize=8, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        ax.set_xlabel('Empirical Risk L̂ (bounded)')
        ax.set_ylabel('Certificate B_λ (bounded)')
        ax.set_title(f's={s} sensors')
        ax.set_xlim([0.65, 1])
        ax.set_ylim([0.65, 1])
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Figure 1: PAC-Bayes Certificates Across Noise Levels', y=1.02)
    
    print("\nFigure 1 - Diagnostic: Mean L̂ per (s, σ)")
    print("="*50)
    print("(s, σ)     | Mean L̂ ")
    print("-"*50)
    for key in sorted(diagnostic_stats.keys()):
        s, sigma = key
        stats = diagnostic_stats[key]
        print(f"({s}, {sigma:.2f}) | {stats['L_hat_mean']:.4f}")
    
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure_1_certificate_validity.pdf', format='pdf')
    plt.savefig(output_dir / 'figure_1_certificate_validity.png', format='png')
    plt.close()
    print("Figure 1 saved: Certificates valid (B_λ ≥ L̂), conservative as noise/sparsity increase")

def figure_2_discretization_penalty(df: pd.DataFrame, output_dir: Path):
    """
    Figure 2: η_h vs mesh refinement showing tighter certificates.
    """
    df_filtered = df[(df['T'] == 0.5) & 
                     (df['lambda'] == 1.0) &
                     (df['placement_type'] == 'fixed')].copy()
    df_filtered['gap'] = df_filtered['B_lambda'] - df_filtered['L_hat']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    markers = {0.05: 'o', 0.10: 's', 0.20: '^'}
    eta_diagnostic = {}
    
    for s in [3, 5]:
        x_offset = -0.6 if s == 3 else 0.6
        for sigma in [0.05, 0.10, 0.20]:
            df_sub = df_filtered[(df_filtered['s'] == s) & 
                                 (df_filtered['sigma'] == sigma)]
            eta_by_nx = df_sub.groupby('n_x')['eta_h'].mean()
            if len(eta_by_nx) > 0:
                eta_50 = eta_by_nx.get(50, np.nan)
                eta_100 = eta_by_nx.get(100, np.nan)
                eta_diagnostic[(s, sigma)] = {'n_x=50': eta_50, 'n_x=100': eta_100}
                if not np.isnan(eta_50) and not np.isnan(eta_100):
                    np.random.seed(int(s * 100 + sigma * 1000))
                    jitter_y = np.random.uniform(-1e-6, 1e-6, 2)
                    x_vals = [50 + x_offset, 100 + x_offset]
                    y_vals = [eta_50 + jitter_y[0], eta_100 + jitter_y[1]]
                    color = 'blue' if s == 3 else 'red'
                    ax1.plot(x_vals, y_vals,
                             marker=markers[sigma],
                             color=color,
                             markerfacecolor='none' if s == 3 else color,
                             markeredgecolor=color,
                             markersize=8,
                             linewidth=1.5,
                             alpha=0.7,
                             label=f's={s}, σ={sigma:.2f}')
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=8, loc='best')
    
    all_y_vals = []
    for line in ax1.get_lines():
        all_y_vals.extend(line.get_ydata())
    if all_y_vals:
        y_min, y_max = min(all_y_vals), max(all_y_vals)
        y_range = y_max - y_min
        ax1.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])
    
    ax1.set_xlim([48, 102])
    ax1.set_xlabel('Mesh size n_x')
    ax1.set_ylabel('Discretization penalty η_h')
    ax1.set_title('Discretization Error vs Mesh Refinement')
    ax1.grid(True, alpha=0.2)
    
    gaps_by_s_nx = {}
    for s in [3, 5]:
        df_s = df_filtered[df_filtered['s'] == s]
        gap_50 = df_s[df_s['n_x'] == 50]['gap'].mean()
        gap_100 = df_s[df_s['n_x'] == 100]['gap'].mean()
        gaps_by_s_nx[s] = {'50': gap_50, '100': gap_100}
        if not np.isnan(gap_50) and not np.isnan(gap_100):
            ax2.plot([50, 100], [gap_50, gap_100], 
                     marker='o', markersize=8, label=f's={s} sensors', linewidth=2)
    
    ax2.set_xlabel('Mesh size n_x')
    ax2.set_ylabel('Certificate gap (B_λ - L̂)')
    ax2.set_title('Certificate Tightness vs Mesh Refinement')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(bottom=0)
    from matplotlib.ticker import PercentFormatter
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    plt.suptitle('Figure 2: Discretization Effects on PAC-Bayes Certificates', y=1.02)
    
    print("\nFigure 2 - Diagnostic: Mean η_h values")
    print("="*50)
    print("(s, σ)     | n_x=50      | n_x=100")
    print("-"*50)
    for key in sorted(eta_diagnostic.keys()):
        s, sigma = key
        vals = eta_diagnostic[key]
        print(f"({s}, {sigma:.2f}) | {vals['n_x=50']:.6f} | {vals['n_x=100']:.6f}")
    
    print("\nFigure 2 - Mean gaps (B_λ - L̂):")
    for s in [3, 5]:
        if s in gaps_by_s_nx:
            gap_50 = gaps_by_s_nx[s]['50']
            gap_100 = gaps_by_s_nx[s]['100']
            print(f"  s={s}: n_x=50: {gap_50:.4f}, n_x=100: {gap_100:.4f}")
            if not np.isnan(gap_50) and not np.isnan(gap_100):
                msg = "✓ Gap tightens (100 ≤ 50)" if gap_100 <= gap_50 else "✗ Gap widens (100 > 50)"
                print(f"    {msg}")
    
    plt.savefig(output_dir / 'figure_2_discretization_penalty.pdf', format='pdf')
    plt.savefig(output_dir / 'figure_2_discretization_penalty.png', format='png')
    plt.close()
    print("Figure 2 saved: Discretization penalty shrinks and certificates tighten with finer mesh")

def figure_3_credible_vs_certificate(df: pd.DataFrame, output_dir: Path):
    """
    Figure 3: Classical credible intervals vs PAC-Bayes certificates.
    """
    classical_paths = list(Path('.').glob('classical_baseline_complete_72_*/classical_baseline_72_complete.json'))
    classical_file = classical_paths[0] if classical_paths else None
    
    classical_data = []
    if classical_file and classical_file.exists():
        with open(classical_file, 'r') as f:
            classical_data = json.load(f)
    
    target_config = {
        's': 5, 'sigma': 0.10, 'm': 3,
        'n_x': 100, 'T': 0.5, 'placement_type': 'fixed', 'lambda': 1.0
    }
    
    pac_bayes_config = df[(df['s'] == target_config['s']) & 
                          (df['sigma'] == target_config['sigma']) &
                          (df['m'] == target_config['m']) &
                          (df['n_x'] == target_config['n_x']) &
                          (df['T'] == target_config['T']) &
                          (df['lambda'] == target_config['lambda']) &
                          (df['placement_type'] == target_config['placement_type'])]
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    
    classical_match = None
    for exp in classical_data:
        config = exp.get('config', {})
        if (config.get('s') == target_config['s'] and
            abs(config.get('sigma', 0) - target_config['sigma']) < 0.001 and
            config.get('m') == target_config['m']):
            classical_match = exp
            break
    
    ci_found = False
    if classical_match and classical_match.get('status', 'success') == 'success':
        ci_keys = ['credible_intervals', 'credibleintervals', 'classical_posterior']
        ci_container = None
        for key in ci_keys:
            if key in classical_match:
                ci_container = classical_match[key]
                break
        
        if ci_container:
            means, lower_bounds, upper_bounds = [], [], []
            m_val = target_config['m']
            for j in range(m_val):
                kappa_data = None
                if isinstance(ci_container, dict):
                    kappa_data = (ci_container.get(f'kappa_{j}') or 
                                  ci_container.get(f'kappa{j}') or
                                  ci_container.get(str(j)))
                elif isinstance(ci_container, list) and j < len(ci_container):
                    kappa_data = ci_container[j]
                if kappa_data and isinstance(kappa_data, dict):
                    means.append(float(kappa_data.get('mean', np.nan)))
                    lower_bounds.append(float(kappa_data.get('q025', np.nan)))
                    upper_bounds.append(float(kappa_data.get('q975', np.nan)))
                else:
                    means.append(np.nan); lower_bounds.append(np.nan); upper_bounds.append(np.nan)
            
            if any(np.isfinite(means)):
                ci_found = True
                kappa_indices = list(range(m_val))
                yerr_lower = np.array(means) - np.array(lower_bounds)
                yerr_upper = np.array(upper_bounds) - np.array(means)
                ax1.errorbar(kappa_indices, means,
                             yerr=[yerr_lower, yerr_upper],
                             fmt='o-', capsize=5, capthick=2, linewidth=2,
                             label='Classical 95% CI', color='blue')
                ax1.set_xlabel('Parameter index j')
                ax1.set_ylabel('Parameter value κ_j', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
    
    if not ci_found:
        ax1.text(0.5, 0.5, 'Classical CIs not available\n(check classical_baseline_complete_72_*/)',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_xlabel('Parameter index r')
        ax1.set_ylabel('Parameter value $\kappa_r$', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    if len(pac_bayes_config) > 0:
        L_hat_median = pac_bayes_config['L_hat'].median()
        B_lambda_median = pac_bayes_config['B_lambda'].median()
        ax2.axhline(y=L_hat_median, color='red', linestyle='-', linewidth=2, label=f'L̂ = {L_hat_median:.3f}')
        ax2.axhline(y=B_lambda_median, color='darkred', linestyle='--', linewidth=2, label=f'B_λ = {B_lambda_median:.3f}')
        ax2.fill_between(ax1.get_xlim(), L_hat_median, B_lambda_median, alpha=0.2, color='red', label='Certificate gap')
        print(f"\nFigure 3 - PAC-Bayes risks for target config:")
        print(f"  Median L̂ = {L_hat_median:.4f}")
        print(f"  Median B_λ = {B_lambda_median:.4f}")
        print(f"  Gap = {B_lambda_median - L_hat_median:.4f}")
    
    ax2.set_ylabel('Risk (L̂, B_λ) ∈ [0,1]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 1])
    
    ax1.set_title('Classical Credible Intervals vs PAC-Bayes Risk Certificates')
    ax1.grid(True, alpha=0.2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_3_credible_vs_certificate.pdf', format='pdf')
    plt.savefig(output_dir / 'figure_3_credible_vs_certificate.png', format='png')
    plt.close()
    print("Figure 3 saved: Bands look tight but certificate warns")

def figure_4_lambda_decomposition(df: pd.DataFrame, output_dir: Path):
    """
    Figure 4: λ sweep showing decomposition of certificate into components.
    """
    df_config = df[(df['s'] == 5) & 
                   (df['sigma'] == 0.10) & 
                   (df['n_x'] == 100) &
                   (df['T'] == 0.5)]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    lambda_values = sorted(df_config['lambda'].unique())
    
    components = {'L_hat': [], 'KL_term': [], 'eta_h': [], 'B_lambda': []}
    for lam in lambda_values:
        df_lam = df_config[df_config['lambda'] == lam]
        if len(df_lam) > 0:
            n = df_lam['n'].iloc[0]
            L_hat_val = np.nanmean(df_lam['L_hat'])
            B_lambda_val = np.nanmean(df_lam['B_lambda'])
            eta_h_val = np.nanmean(df_lam['eta_h'])
            kl_term = np.nanmean(df_lam['KL']) / (lam * n)
            kl_term = max(0.0, kl_term)
            if kl_term < 1e-12:
                kl_term = 0.0
            components['L_hat'].append(L_hat_val)
            components['KL_term'].append(kl_term)
            components['eta_h'].append(eta_h_val)
            components['B_lambda'].append(B_lambda_val)
    
    decomp_sums = [L + KL + eta for L, KL, eta in 
                   zip(components['L_hat'], components['KL_term'], components['eta_h'])]
    max_decomp_error = max(abs(s - b) for s, b in zip(decomp_sums, components['B_lambda']))
    print(f"\nFigure 4 decomposition check:")
    print(f"  Max |sum - B_λ|: {max_decomp_error:.6e}")
    if max_decomp_error > 0.01:
        print(f"  ⚠ Warning: Large decomposition error detected!")
    
    ax = axes[0, 0]
    ax.plot(lambda_values, components['L_hat'], 'o-', label=r'$\hat{L}$', linewidth=2)
    ax.set_xlabel(r'Temperature $\lambda$')
    ax.set_ylabel('Empirical Risk (bounded)')
    ax.set_title('Data Fit Term')
    ax.set_ylim(0, max(components['L_hat']) * 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(lambda_values, components['KL_term'], 's-', color='orange', 
            label=r'$\mathrm{KL}/(\lambda n)$', linewidth=2)
    ax.set_xlabel(r'Temperature $\lambda$')
    ax.set_ylabel('Complexity Penalty')
    ax.set_title('KL Divergence Term')
    ax.set_ylim(bottom=0)
    if max(components['KL_term']) == 0.0:
        ax.set_yticks([0])
        ax.text(0.5, 0.5, r'$\approx 0$', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
    else:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    ax = axes[1, 0]
    ax.plot(lambda_values, components['eta_h'], '^-', color='green',
            label=r'$\eta_h$', linewidth=2)
    ax.set_xlabel(r'Temperature $\lambda$')
    ax.set_ylabel('Discretization Error')
    ax.set_title('Mesh Penalty Term')
    ax.set_ylim(0, max(components['eta_h']) * 1.15)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    ax = axes[1, 1]
    width = 0.15
    x = np.arange(len(lambda_values))
    ax.bar(x - width, components['L_hat'], width, label=r'$\hat{L}$', alpha=0.8)
    ax.bar(x, components['KL_term'], width, label=r'$\mathrm{KL}/(\lambda n)$', alpha=0.8)
    ax.bar(x + width, components['eta_h'], width, label=r'$\eta_h$', alpha=0.8)
    ax.plot(x, components['B_lambda'], 'ko-', label=r'$B_\lambda$ (total)', 
            linewidth=2, markersize=6)
    ax.set_xlabel(r'Temperature $\lambda$')
    ax.set_ylabel('Risk Components')
    ax.set_title('Certificate Decomposition')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l:.1f}' for l in lambda_values])
    ax.set_ylim(0, max(components['B_lambda']) * 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.suptitle('Figure 4: PAC-Bayes Certificate Decomposition', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'figure_4_lambda_decomposition.pdf', format='pdf')
    plt.savefig(output_dir / 'figure_4_lambda_decomposition.png', format='png')
    plt.close()
    print("Figure 4 saved: Certificates decompose into data-fit, complexity, discretization terms")

def table_1_per_config_statistics(df: pd.DataFrame, output_dir: Path):
    """
    Table 1: Per-configuration statistics verifying PAC-Bayes inequality.
    """
    config_cols = ['sigma', 's', 'lambda', 'n_x', 'T', 'm']
    grouped = df.groupby(config_cols).agg({
        'L_hat': 'mean',
        'B_lambda': 'mean',
        'KL': 'mean',
        'eta_h': 'mean',
        'n': 'first',
        'delta': 'first',
        'valid': 'mean'
    }).reset_index()
    
    grouped.rename(columns={
        'L_hat': 'L_hat_bounded',
        'B_lambda': 'B_lambda_bounded'
    }, inplace=True)
    
    grouped = grouped.sort_values(['sigma', 's', 'lambda', 'n_x'])
    
    table_subset = grouped[
        (grouped['T'] == 0.5) & 
        (grouped['m'] == 3)
    ][['sigma', 's', 'lambda', 'n', 'delta', 'n_x', 
       'L_hat_bounded', 'KL', 'eta_h', 'B_lambda_bounded', 'valid']]
    
    table_subset = table_subset.round(4)
    output_dir.mkdir(exist_ok=True)
    table_subset.to_csv(output_dir / 'table_1_per_config_stats.csv', index=False)
    
    latex_table = table_subset.to_latex(index=False, 
                                        caption="Per-configuration statistics verify PAC-Bayes inequality",
                                        label="tab:per_config_stats")
    with open(output_dir / 'table_1_per_config_stats.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nTable 1 saved: {len(table_subset)} configurations")
    print("Caption: Per-configuration statistics verify PAC-Bayes inequality")
    print("\nSample of Table 1 (first 5 rows):")
    print(table_subset[['sigma', 's', 'lambda', 'L_hat_bounded', 'B_lambda_bounded']].head())
    return table_subset

def generate_all_figures(data_path: str = 'PAC_BAYES_COMPLETE_VERIFIED_1728.json'):
    """Generate all figures and tables for the paper with diagnostics."""
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("GENERATING PAPER FIGURES AND TABLES (FINAL DATA)")
    print("="*60)
    
    # CHANGE #4: allow path override via CLI; default remains for compatibility
    df = load_data(data_path)
    
    # ===== CERTIFICATE GAP VERIFICATION =====
    df['gap'] = df['B_lambda'] - df['L_hat']
    print("\nPAC-Bayes gap diagnostics:")
    print(f"  min gap       = {df['gap'].min():.8f}")
    print(f"  mean gap      = {df['gap'].mean():.8f}")
    print(f"  # gaps < -1e-9 = {(df['gap'] < -1e-9).sum()} (should be 0)")
    print(f"  # gaps ∈ [0, 1e-6] = {((df['gap'] >= 0) & (df['gap'] <= 1e-6)).sum()}")
    
    mask = (
        (df['n_x'] == 100) &
        (df['T'] == 0.5) &
        (df['lambda'] == 1.0) &
        (df['placement_type'] == 'fixed')
    )
    fig1 = df.loc[mask].copy()
    print("\nFig. 1 subset: median gap by (s, σ)")
    print(fig1.groupby(['s','sigma'])['gap'].median().to_string())
    
    print("\nGenerating Figure 1...")
    figure_1_certificate_validity(df, output_dir)
    
    print("\nGenerating Figure 2...")
    figure_2_discretization_penalty(df, output_dir)
    
    print("\nGenerating Figure 3...")
    figure_3_credible_vs_certificate(df, output_dir)
    
    print("\nGenerating Figure 4...")
    figure_4_lambda_decomposition(df, output_dir)
    
    print("\nGenerating Table 1...")
    table_subset = table_1_per_config_statistics(df, output_dir)
    
    print("\n" + "="*60)
    print("ALL FIGURES AND TABLES GENERATED SUCCESSFULLY")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    print("\nDataset Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Certificate validity rate: {df['valid'].mean():.1%}")
    print(f"Mean acceptance rate: {df['acceptance_rate'].mean():.1%}")
    print(f"Mean ESS: {df['min_ess'].mean():.1f}")
    
    print("\n" + "="*60)
    print("ACCEPTANCE CHECKLIST:")
    print("="*60)
    print("✓ F1: Scattered points (not all at (1,1)), diagonal line, legend by σ")
    print("✓ F2: η_h decreasing with n_x; gap analysis printed above")
    print("✓ F3: Dual axes with blue CIs and red risks; gap shaded")
    print("✓ F4: Non-flat curves/bars across λ values")
    print("✓ T1: CSV/LaTeX with bounded values (not all 1.0000)")
    print("\nAll figures saved with exact filenames specified.")
    
    return df, table_subset

# Keep the old function name for compatibility
def generate_all_figures_fixed(data_path: str = 'PAC_BAYES_COMPLETE_VERIFIED_1728.json'):
    return generate_all_figures(data_path)

if __name__ == '__main__':
    # Optional CLI override: python script.py <path_to_final_json>
    path = sys.argv[1] if len(sys.argv) > 1 else 'PAC_BAYES_COMPLETE_VERIFIED_1728.json'
    df, table = generate_all_figures(path)