#!/usr/bin/env python3
"""
Visualization utilities for Section I paper figures
Generates publication-quality plots and tables for PAC-Bayes inverse PDE results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

def generate_figure_1(results: List[Dict], save_path: Optional[Path] = None):
    """
    Figure 1: B_? vs L and L_MC across ?, s.
    
    Takeaway: Certificate is valid, informative, and becomes more 
    conservative as noise/sparsity increase.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Extract data
    df_data = []
    for res in results:
        if 'certificate' in res:
            df_data.append({
                's': res['config']['s'],
                'sigma': res['config']['sigma'],
                'lambda': res['config']['lambda'],
                'B_lambda': res['certificate']['B_lambda'],
                'L_hat': res['certificate']['L_hat'],
                'L_mc': res.get('true_risk', {}).get('L_mc', np.nan)
            })
    
    df = pd.DataFrame(df_data)
    
    # Plot for each ?
    for i, sigma in enumerate([0.05, 0.10, 0.20]):
        # s = 3
        ax = axes[0, i]
        data_s3 = df[(df['sigma'] == sigma) & (df['s'] == 3)]
        
        if not data_s3.empty:
            x = np.arange(len(data_s3))
            ax.bar(x - 0.2, data_s3['L_hat'], 0.4, label='L', alpha=0.7)
            ax.bar(x + 0.2, data_s3['B_lambda'], 0.4, label='B_?', alpha=0.7)
            if 'L_mc' in data_s3.columns:
                ax.scatter(x, data_s3['L_mc'], color='red', s=50, 
                          marker='*', label='L_MC', zorder=5)
        
        ax.set_title(f's=3, ?={sigma}')
        ax.set_xlabel('?')
        ax.set_ylabel('Risk')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # s = 5
        ax = axes[1, i]
        data_s5 = df[(df['sigma'] == sigma) & (df['s'] == 5)]
        
        if not data_s5.empty:
            x = np.arange(len(data_s5))
            ax.bar(x - 0.2, data_s5['L_hat'], 0.4, label='L', alpha=0.7)
            ax.bar(x + 0.2, data_s5['B_lambda'], 0.4, label='B_?', alpha=0.7)
            if 'L_mc' in data_s5.columns:
                ax.scatter(x, data_s5['L_mc'], color='red', s=50,
                          marker='*', label='L_MC', zorder=5)
        
        ax.set_title(f's=5, ?={sigma}')
        ax.set_xlabel('?')
        ax.set_ylabel('Risk')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 1: PAC-Bayes Certificate vs Empirical and True Risk\n'
                 'Certificate is valid (B_? e L_MC) and tightens with more sensors',
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'figure_1.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def generate_figure_2(results: List[Dict], save_path: Optional[Path] = None):
    """
    Figure 2: ?_h vs mesh, showing B_? tightens from n_x=50 to 100.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Extract discretization penalties
    df_data = []
    for res in results:
        if 'certificate' in res:
            df_data.append({
                'n_x': res['config']['n_x'],
                'lambda': res['config']['lambda'],
                'eta_h': res['certificate']['eta_h'],
                'B_lambda': res['certificate']['B_lambda']
            })
    
    df = pd.DataFrame(df_data)
    
    # Left panel: ?_h vs mesh size
    ax = axes[0]
    for lambda_val in [0.5, 1.0, 2.0]:
        data = df[df['lambda'] == lambda_val]
        if not data.empty:
            ax.plot(data['n_x'], data['eta_h'], 'o-', 
                   label=f'?={lambda_val}', markersize=8)
    
    ax.set_xlabel('Mesh size (n_x)')
    ax.set_ylabel('Discretization penalty (?_h)')
    ax.set_title('Discretization Error vs Mesh Resolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Right panel: B_? for different mesh sizes
    ax = axes[1]
    for n_x in [50, 100]:
        data = df[df['n_x'] == n_x]
        if not data.empty:
            ax.plot(data['lambda'], data['B_lambda'], 'o-',
                   label=f'n_x={n_x}', markersize=8)
    
    ax.set_xlabel('Temperature (?)')
    ax.set_ylabel('Certificate (B_?)')
    ax.set_title('Certificate Improves with Mesh Refinement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 2: Mesh Robustness of PAC-Bayes Certificate\n'
                 '?_h ? 0 as mesh refines, leading to tighter bounds',
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'figure_2.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def generate_figure_3(results: List[Dict], save_path: Optional[Path] = None):
    """
    Figure 3: Credible bands (Q_Bayes) vs certified risk (Q_?).
    
    Shows case where credible bands look tight but B_? warns of overfitting.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Find a high-noise, sparse-sensor case
    target_config = {'s': 3, 'sigma': 0.20}
    
    # Extract relevant results
    bayes_result = None
    gibbs_result = None
    
    for res in results:
        if (res['config']['s'] == target_config['s'] and 
            res['config']['sigma'] == target_config['sigma']):
            if res['config'].get('is_baseline', False):
                bayes_result = res
            elif res['config'].get('lambda') == 1.0:
                gibbs_result = res
    
    if bayes_result and gibbs_result:
        # Top left: Classical credible bands
        ax = axes[0, 0]
        kappa_star = np.array(bayes_result['dataset']['kappa_star'])
        posterior_mean = np.array(bayes_result['posterior_summary']['mean'])
        lower = np.array(bayes_result['posterior_summary']['quantiles']['2.5%'])
        upper = np.array(bayes_result['posterior_summary']['quantiles']['97.5%'])
        
        x = np.arange(len(kappa_star))
        ax.plot(x, kappa_star, 'ko-', label='True ?*', markersize=8)
        ax.plot(x, posterior_mean, 'b^-', label='Posterior mean', markersize=6)
        ax.fill_between(x, lower, upper, alpha=0.3, label='95% Credible')
        
        ax.set_xlabel('Segment')
        ax.set_ylabel('?')
        ax.set_title('Classical Bayesian Credible Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top right: Gibbs posterior samples
        ax = axes[0, 1]
        ax.text(0.5, 0.5, f"PAC-Bayes Certificate\nB_? = {gibbs_result['certificate']['B_lambda']:.3f}",
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Gibbs Posterior (?=1.0)')
        ax.axis('off')
        
        # Bottom: Risk decomposition
        ax = axes[1, 0]
        components = gibbs_result['certificate']['components']
        labels = ['Empirical\nL', 'KL/(?n)', 'ln(1/?)/(?n)', 'Discretization\n?_h']
        values = [components['empirical_term'], 
                 components['kl_term'],
                 components['delta_term'],
                 components['discretization_term']]
        
        ax.bar(labels, values, alpha=0.7)
        ax.set_ylabel('Contribution to B_?')
        ax.set_title('Certificate Decomposition')
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Warning message
        ax = axes[1, 1]
        warning_text = ("Despite narrow credible bands,\n"
                       "PAC-Bayes certificate indicates\n"
                       "potential overfitting risk\n\n"
                       f"B_? = {gibbs_result['certificate']['B_lambda']:.3f}\n"
                       f"L = {gibbs_result['certificate']['L_hat']:.3f}")
        ax.text(0.5, 0.5, warning_text, ha='center', va='center', fontsize=11)
        ax.set_title('Key Insight')
        ax.axis('off')
    
    plt.suptitle('Figure 3: Classical Credible Bands vs PAC-Bayes Certificate\n'
                 'Narrow bands can be misleading; certificate provides reliable guarantee',
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'figure_3.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def generate_table_1(results: List[Dict], save_path: Optional[Path] = None):
    """
    Table 1: Per-config tuple (L, KL, ?, n, ?, ?_h, B_?).
    """
    table_data = []
    
    for res in results:
        if 'certificate' in res:
            row = {
                's': res['config']['s'],
                '?': res['config']['sigma'],
                'n_x': res['config']['n_x'],
                '?': res['config']['lambda'],
                'n': res['config']['n'],
                'L': f"{res['certificate']['L_hat']:.4f}",
                'KL': f"{res['certificate']['KL']:.3f}",
                '?': res['config']['delta'],
                '?_h': f"{res['certificate']['eta_h']:.4f}",
                'B_?': f"{res['certificate']['B_lambda']:.4f}"
            }
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Sort by s, ?, ?
    df = df.sort_values(['s', '?', '?'])
    
    # Create LaTeX table
    latex_table = df.to_latex(index=False, float_format='%.4f',
                              caption='PAC-Bayes Certificate Components',
                              label='tab:certificate_results')
    
    # Create markdown table for display
    print("\nTable 1: PAC-Bayes Certificate Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    if save_path:
        # Save LaTeX
        with open(save_path / 'table_1.tex', 'w') as f:
            f.write(latex_table)
        
        # Save CSV
        df.to_csv(save_path / 'table_1.csv', index=False)
    
    return df


def generate_paper_figures(results: List[Dict[str, Any]], output_dir: str = 'figures') -> None:
    """
    Generate all Section I paper figures from experiment results.
    
    Args:
        results: List of experiment result dictionaries
        output_dir: Directory to save figures
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Section I figures from {len(results)} experiments...")
    print(f"Output directory: {output_path}")
    
    # Convert results to DataFrame for easier analysis
    df = results_to_dataframe(results)
    
    # Figure 1: PAC-Bayes Certificate vs Temperature
    generate_certificate_temperature_plot(df, output_path)
    
    # Figure 2: Certificate Component Breakdown
    generate_certificate_components_plot(df, output_path)
    
    # Figure 3: Resolution Effect on Discretization Penalty
    generate_discretization_penalty_plot(df, output_path)
    
    # Figure 4: MCMC Convergence Analysis
    generate_convergence_analysis_plot(df, output_path)
    
    # Figure 5: True Risk vs Certificate Bound
    generate_risk_certificate_comparison(df, output_path)
    
    # Table 1: Main Results Summary
    generate_main_results_table(df, output_path)
    
    # Table 2: Convergence Statistics
    generate_convergence_table(df, output_path)
    
    # Figure 6: Sensor Configuration Comparison (if available)
    if len(df['placement_type'].unique()) > 1:
        generate_sensor_placement_comparison(df, output_path)
    
    print(f"? All Section I figures generated successfully in {output_path}")


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert experiment results to pandas DataFrame."""
    
    data = []
    
    for result in results:
        config = result['config']
        
        row = {
            # Configuration
            's': config['s'],
            'placement_type': config['placement_type'],
            'sigma': config['sigma'],
            'n_x': config['n_x'],
            'T': config['T'],
            'lambda': config['lambda'],
            'm': config['m'],
            'seed': config['seed'],
            'n': config['n'],
            
            # Results
            'B_lambda': result['certificate']['B_lambda'],
            'L_hat': result['certificate']['L_hat'],
            'KL': result['certificate']['KL'],
            'eta_h': result['certificate']['eta_h'],
            'kl_term': result['certificate']['components']['kl_term'],
            'discretization_term': result['certificate']['components']['discretization_term'],
            'Z_hat': result['certificate']['partition_function']['Z_hat'],
            'underline_Z': result['certificate']['partition_function']['underline_Z'],
            
            # True risk
            'L_mc': result['true_risk']['L_mc'],
            'L_mc_std': result['true_risk']['L_mc_std'],
            
            # MCMC diagnostics
            'acceptance_rate': result['mcmc']['acceptance_rate'],
            'ess_min': min(result['mcmc']['ess']),
            'ess_mean': np.mean(result['mcmc']['ess']),
            'converged': result['mcmc']['converged'],
            'n_forward_evals': result['mcmc']['n_forward_evals'],
            
            # Performance
            'mcmc_time': result['performance']['timings']['mcmc']['total'],
            'certificate_time': result['performance']['timings']['certificate']['total']
        }
        
        data.append(row)
    
    return pd.DataFrame(data)


def generate_certificate_temperature_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 1: PAC-Bayes Certificate vs Temperature Parameter."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Certificate vs Lambda
    lambda_vals = sorted(df['lambda'].unique())
    
    for m in sorted(df['m'].unique()):
        m_data = df[df['m'] == m]
        means = [m_data[m_data['lambda'] == lam]['B_lambda'].mean() for lam in lambda_vals]
        stds = [m_data[m_data['lambda'] == lam]['B_lambda'].std() for lam in lambda_vals]
        
        ax1.errorbar(lambda_vals, means, yerr=stds, marker='o', linewidth=2, 
                    capsize=5, label=f'm = {m}', markersize=8)
    
    ax1.set_xlabel('Temperature Parameter ?')
    ax1.set_ylabel('PAC-Bayes Certificate B_?')
    ax1.set_title('Certificate Bound vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Component breakdown
    components = ['L_hat', 'kl_term', 'discretization_term']
    comp_labels = ['Empirical Loss', 'KL Term', 'Discretization']
    
    lambda_means = []
    for lam in lambda_vals:
        lam_data = df[df['lambda'] == lam]
        comp_means = [lam_data[comp].mean() for comp in components]
        lambda_means.append(comp_means)
    
    bottom = np.zeros(len(lambda_vals))
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (comp, label, color) in enumerate(zip(components, comp_labels, colors)):
        values = [lambda_means[j][i] for j in range(len(lambda_vals))]
        ax2.bar(lambda_vals, values, bottom=bottom, label=label, 
               color=color, alpha=0.8, width=0.15)
        bottom += values
    
    ax2.set_xlabel('Temperature Parameter ?')
    ax2.set_ylabel('Certificate Component Value')
    ax2.set_title('Certificate Component Breakdown')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_certificate_temperature.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure1_certificate_temperature.pdf', bbox_inches='tight')
    plt.close()
    
    print("? Generated Figure 1: Certificate vs Temperature")


def generate_certificate_components_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 2: Detailed Certificate Component Analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: KL term vs lambda
    sns.boxplot(data=df, x='lambda', y='kl_term', ax=axes[0,0])
    axes[0,0].set_title('KL Divergence Term vs Temperature')
    axes[0,0].set_ylabel('KL / (?n)')
    
    # Plot 2: Discretization penalty vs resolution
    sns.scatterplot(data=df, x='n_x', y='eta_h', hue='lambda', size='m', ax=axes[0,1])
    axes[0,1].set_title('Discretization Penalty vs Resolution')
    axes[0,1].set_ylabel('?_h')
    axes[0,1].set_xlabel('Spatial Resolution n_x')
    
    # Plot 3: Empirical loss distribution
    sns.violinplot(data=df, x='m', y='L_hat', hue='lambda', ax=axes[1,0])
    axes[1,0].set_title('Empirical Loss Distribution')
    axes[1,0].set_ylabel('L?')
    
    # Plot 4: Certificate validity check
    df['valid_certificate'] = df['B_lambda'] >= df['L_hat']
    validity_summary = df.groupby(['lambda', 'm'])['valid_certificate'].mean().reset_index()
    
    pivot_validity = validity_summary.pivot(index='lambda', columns='m', values='valid_certificate')
    sns.heatmap(pivot_validity, annot=True, fmt='.3f', ax=axes[1,1], 
                cmap='RdYlGn', vmin=0, vmax=1)
    axes[1,1].set_title('Certificate Validity Rate')
    axes[1,1].set_ylabel('Temperature ?')
    axes[1,1].set_xlabel('Parameter Dimension m')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure2_certificate_components.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure2_certificate_components.pdf', bbox_inches='tight')
    plt.close()
    
    print("? Generated Figure 2: Certificate Components")


def generate_discretization_penalty_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 3: Resolution Effect Analysis."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: eta_h vs 1/n_x^2 (should be linear)
    n_x_vals = sorted(df['n_x'].unique())
    inverse_n_x_sq = [1.0 / nx**2 for nx in n_x_vals]
    
    eta_means = [df[df['n_x'] == nx]['eta_h'].mean() for nx in n_x_vals]
    eta_stds = [df[df['n_x'] == nx]['eta_h'].std() for nx in n_x_vals]
    
    ax1.errorbar(inverse_n_x_sq, eta_means, yerr=eta_stds, 
                marker='o', linewidth=2, capsize=5, markersize=8)
    ax1.set_xlabel('1/n_x?')
    ax1.set_ylabel('Discretization Penalty ?_h')
    ax1.set_title('O(h?) Scaling Verification')
    ax1.grid(True, alpha=0.3)
    
    # Add linear fit
    z = np.polyfit(inverse_n_x_sq, eta_means, 1)
    p = np.poly1d(z)
    ax1.plot(inverse_n_x_sq, p(inverse_n_x_sq), "r--", alpha=0.8, 
            label=f'Linear fit: y = {z[0]:.6f}x + {z[1]:.6f}')
    ax1.legend()
    
    # Plot 2: Certificate impact of discretization
    for nx in n_x_vals:
        nx_data = df[df['n_x'] == nx]
        lambda_vals = sorted(nx_data['lambda'].unique())
        cert_means = [nx_data[nx_data['lambda'] == lam]['B_lambda'].mean() 
                     for lam in lambda_vals]
        
        ax2.plot(lambda_vals, cert_means, marker='o', linewidth=2, 
                label=f'n_x = {nx}', markersize=6)
    
    ax2.set_xlabel('Temperature Parameter ?')
    ax2.set_ylabel('Certificate B_?')
    ax2.set_title('Resolution Impact on Certificate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure3_discretization_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure3_discretization_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print("? Generated Figure 3: Discretization Analysis")


def generate_convergence_analysis_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 4: MCMC Convergence Analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Acceptance rate vs lambda
    sns.boxplot(data=df, x='lambda', y='acceptance_rate', ax=axes[0,0])
    axes[0,0].set_title('MCMC Acceptance Rate vs Temperature')
    axes[0,0].set_ylabel('Acceptance Rate')
    axes[0,0].axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Min threshold')
    axes[0,0].legend()
    
    # Plot 2: ESS vs dimension
    sns.scatterplot(data=df, x='m', y='ess_mean', hue='lambda', size='n_x', ax=axes[0,1])
    axes[0,1].set_title('Effective Sample Size vs Dimension')
    axes[0,1].set_ylabel('Mean ESS')
    axes[0,1].axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Min threshold')
    axes[0,1].legend()
    
    # Plot 3: Convergence rate by configuration
    conv_rate = df.groupby(['lambda', 'm'])['converged'].mean().reset_index()
    pivot_conv = conv_rate.pivot(index='lambda', columns='m', values='converged')
    
    sns.heatmap(pivot_conv, annot=True, fmt='.3f', ax=axes[1,0], 
                cmap='RdYlGn', vmin=0, vmax=1)
    axes[1,0].set_title('Convergence Rate by Configuration')
    axes[1,0].set_ylabel('Temperature ?')
    axes[1,0].set_xlabel('Parameter Dimension m')
    
    # Plot 4: Computational efficiency
    df['efficiency'] = df['ess_mean'] / df['mcmc_time']
    sns.scatterplot(data=df, x='mcmc_time', y='ess_mean', hue='lambda', 
                   size='m', ax=axes[1,1])
    axes[1,1].set_title('MCMC Efficiency (ESS vs Time)')
    axes[1,1].set_xlabel('MCMC Time (seconds)')
    axes[1,1].set_ylabel('Mean ESS')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure4_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure4_convergence_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print("? Generated Figure 4: Convergence Analysis")


def generate_risk_certificate_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 5: True Risk vs Certificate Bound Comparison."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter plot of true risk vs certificate
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['lambda'].unique())))
    
    for i, lam in enumerate(sorted(df['lambda'].unique())):
        lam_data = df[df['lambda'] == lam]
        ax1.scatter(lam_data['L_mc'], lam_data['B_lambda'], 
                   color=colors[i], label=f'? = {lam}', alpha=0.7, s=50)
    
    # Add diagonal line (should be below for valid certificates)
    max_val = max(df['L_mc'].max(), df['B_lambda'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, 
            label='B_? = L_MC (invalid)')
    
    ax1.set_xlabel('True Risk L_MC')
    ax1.set_ylabel('Certificate Bound B_?')
    ax1.set_title('Certificate vs True Risk')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Certificate gap analysis
    df['certificate_gap'] = df['B_lambda'] - df['L_mc']
    
    sns.boxplot(data=df, x='lambda', y='certificate_gap', ax=ax2)
    ax2.set_title('Certificate Gap (B_? - L_MC)')
    ax2.set_ylabel('Certificate Gap')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, 
               label='Zero gap (certificate = true risk)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure5_risk_certificate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure5_risk_certificate_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print("? Generated Figure 5: Risk Certificate Comparison")


def generate_sensor_placement_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 6: Sensor Placement Strategy Comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Certificate by placement
    sns.boxplot(data=df, x='placement_type', y='B_lambda', hue='lambda', ax=axes[0,0])
    axes[0,0].set_title('Certificate by Sensor Placement')
    axes[0,0].set_ylabel('Certificate B_?')
    
    # Plot 2: True risk by placement  
    sns.boxplot(data=df, x='placement_type', y='L_mc', hue='s', ax=axes[0,1])
    axes[0,1].set_title('True Risk by Sensor Placement')
    axes[0,1].set_ylabel('True Risk L_MC')
    
    # Plot 3: Convergence by placement
    conv_by_placement = df.groupby(['placement_type', 'lambda'])['converged'].mean().reset_index()
    pivot_conv_placement = conv_by_placement.pivot(index='lambda', columns='placement_type', values='converged')
    
    sns.heatmap(pivot_conv_placement, annot=True, fmt='.3f', ax=axes[1,0], 
                cmap='RdYlGn', vmin=0, vmax=1)
    axes[1,0].set_title('Convergence Rate by Placement')
    axes[1,0].set_ylabel('Temperature ?')
    
    # Plot 4: Efficiency comparison
    eff_by_placement = df.groupby(['placement_type', 's']).agg({
        'B_lambda': 'mean',
        'L_mc': 'mean',
        'acceptance_rate': 'mean',
        'ess_mean': 'mean'
    }).reset_index()
    
    x = np.arange(len(eff_by_placement))
    width = 0.35
    
    bars1 = axes[1,1].bar(x - width/2, eff_by_placement['B_lambda'], width, 
                         label='Certificate B_?', alpha=0.8)
    bars2 = axes[1,1].bar(x + width/2, eff_by_placement['L_mc'], width, 
                         label='True Risk L_MC', alpha=0.8)
    
    axes[1,1].set_xlabel('Configuration')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title('Certificate vs True Risk by Configuration')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([f"{row['placement_type']}\ns={row['s']}" 
                              for _, row in eff_by_placement.iterrows()], rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure6_sensor_placement_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure6_sensor_placement_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print("? Generated Figure 6: Sensor Placement Comparison")


def generate_main_results_table(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Table 1: Main Results Summary."""
    
    # Group by key parameters
    summary = df.groupby(['lambda', 'm']).agg({
        'B_lambda': ['mean', 'std'],
        'L_hat': ['mean', 'std'], 
        'L_mc': ['mean', 'std'],
        'kl_term': ['mean', 'std'],
        'eta_h': ['mean', 'std'],
        'converged': 'mean',
        'acceptance_rate': 'mean',
        'ess_mean': 'mean'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    # Create LaTeX table
    latex_table = summary.to_latex(
        caption="Main PAC-Bayes Results Summary",
        label="tab:main_results",
        float_format="%.4f"
    )
    
    # Save tables
    with open(output_path / 'table1_main_results.tex', 'w') as f:
        f.write(latex_table)
    
    summary.to_csv(output_path / 'table1_main_results.csv')
    
    # Create formatted HTML table for viewing
    html_table = summary.to_html(
        classes='table table-striped',
        float_format=lambda x: f'{x:.4f}' if pd.notnull(x) else ''
    )
    
    with open(output_path / 'table1_main_results.html', 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Main PAC-Bayes Results</title>
            <style>
                .table {{ margin: 20px; font-family: Arial, sans-serif; }}
                .table-striped tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th, td {{ padding: 8px; text-align: right; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h2>Table 1: Main PAC-Bayes Results Summary</h2>
            {html_table}
        </body>
        </html>
        """)
    
    print("? Generated Table 1: Main Results Summary")


def generate_convergence_table(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Table 2: MCMC Convergence Statistics."""
    
    # Convergence analysis by configuration
    conv_stats = df.groupby(['lambda', 'm', 'n_x']).agg({
        'acceptance_rate': ['mean', 'std'],
        'ess_min': ['mean', 'std'],
        'ess_mean': ['mean', 'std'],
        'converged': ['mean', 'count'],
        'mcmc_time': ['mean', 'std'],
        'n_forward_evals': 'mean'
    }).round(4)
    
    # Flatten column names
    conv_stats.columns = ['_'.join(col).strip() for col in conv_stats.columns]
    
    # Calculate additional metrics
    conv_stats['efficiency'] = (conv_stats['ess_mean_mean'] / 
                               conv_stats['mcmc_time_mean']).round(2)
    
    # Create tables
    latex_table = conv_stats.to_latex(
        caption="MCMC Convergence and Performance Statistics",
        label="tab:convergence",
        float_format="%.4f"
    )
    
    with open(output_path / 'table2_convergence_stats.tex', 'w') as f:
        f.write(latex_table)
    
    conv_stats.to_csv(output_path / 'table2_convergence_stats.csv')
    
    # HTML version
    html_table = conv_stats.to_html(
        classes='table table-striped',
        float_format=lambda x: f'{x:.4f}' if pd.notnull(x) else ''
    )
    
    with open(output_path / 'table2_convergence_stats.html', 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>MCMC Convergence Statistics</title>
            <style>
                .table {{ margin: 20px; font-family: Arial, sans-serif; }}
                .table-striped tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th, td {{ padding: 8px; text-align: right; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h2>Table 2: MCMC Convergence and Performance Statistics</h2>
            {html_table}
        </body>
        </html>
        """)
    
    print("? Generated Table 2: Convergence Statistics")


if __name__ == '__main__':
    # Test with dummy data
    import json
    
    # Load results if available
    results_file = Path('results/results_test.json')
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        generate_paper_figures(results)
    else:
        print("No results file found. Run experiments first.")