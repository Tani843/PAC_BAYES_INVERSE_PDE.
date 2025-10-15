#!/usr/bin/env python3
"""
Analyze key findings from complete Section A results
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    """Analyze key findings from the complete experimental results."""
    
    print("=" * 80)
    print("ANALYZING KEY FINDINGS FROM SECTION A RESULTS")
    print("=" * 80)
    
    # Load results
    with open('results_full_section_a/section_a_complete_20250917_142211.json', 'r') as f:
        results = json.load(f)
    
    print(f"Analyzing {len(results)} experiments...")
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame([r['config'] for r in results])
    df['B_lambda'] = [r['certificate']['B_lambda'] for r in results]
    df['L_hat'] = [r['certificate']['L_hat'] for r in results]
    df['L_mc'] = [r['true_risk']['L_mc'] for r in results]
    df['KL'] = [r['certificate']['KL'] for r in results]
    df['eta_h'] = [r['certificate']['eta_h'] for r in results]
    df['acceptance_rate'] = [r['mcmc']['acceptance_rate'] for r in results]
    df['converged'] = [r['mcmc']['converged'] for r in results]
    
    print("\n" + "=" * 60)
    print("📊 KEY FINDINGS ANALYSIS")
    print("=" * 60)
    
    # 1. Certificate validity analysis
    validity_rate = (df['B_lambda'] >= df['L_mc']).mean()
    print(f"\n1. CERTIFICATE VALIDITY:")
    print(f"   • Certificate validity rate (B_λ ≥ L_MC): {validity_rate:.3f}")
    print(f"   • Valid certificates: {(df['B_lambda'] >= df['L_mc']).sum()}/{len(df)}")
    
    # 2. Temperature parameter analysis
    print(f"\n2. TEMPERATURE PARAMETER EFFECTS:")
    lambda_analysis = df.groupby('lambda').agg({
        'B_lambda': ['mean', 'std'],
        'L_hat': 'mean',
        'L_mc': 'mean',
        'KL': 'mean'
    }).round(4)
    
    print("   Best performing λ (lowest B_λ):")
    lambda_means = df.groupby('lambda')['B_lambda'].mean()
    best_lambda = lambda_means.idxmin()
    print(f"   • λ = {best_lambda}: B_λ = {lambda_means[best_lambda]:.4f}")
    
    print(f"\n   Temperature parameter effects:")
    for lam in sorted(df['lambda'].unique()):
        lam_data = df[df['lambda'] == lam]
        print(f"   • λ = {lam}: B_λ = {lam_data['B_lambda'].mean():.4f} ± {lam_data['B_lambda'].std():.4f}")
    
    # 3. Noise level analysis
    print(f"\n3. NOISE LEVEL EFFECTS:")
    noise_effects = df.groupby('sigma')['B_lambda'].agg(['mean', 'std']).round(4)
    print("   Effect of noise on certificate:")
    for sigma in sorted(df['sigma'].unique()):
        sigma_data = df[df['sigma'] == sigma]
        print(f"   • σ = {sigma}: B_λ = {sigma_data['B_lambda'].mean():.4f} ± {sigma_data['B_lambda'].std():.4f}")
    
    # 4. Sensor configuration analysis
    print(f"\n4. SENSOR CONFIGURATION EFFECTS:")
    sensor_effects = df.groupby('s')['B_lambda'].agg(['mean', 'std']).round(4)
    print("   Effect of sensor count:")
    for s in sorted(df['s'].unique()):
        s_data = df[df['s'] == s]
        print(f"   • s = {s} sensors: B_λ = {s_data['B_lambda'].mean():.4f} ± {s_data['B_lambda'].std():.4f}")
    
    # 5. Placement strategy analysis
    print(f"\n5. SENSOR PLACEMENT STRATEGY:")
    placement_effects = df.groupby('placement_type')['B_lambda'].agg(['mean', 'std']).round(4)
    for placement in sorted(df['placement_type'].unique()):
        placement_data = df[df['placement_type'] == placement]
        print(f"   • {placement} placement: B_λ = {placement_data['B_lambda'].mean():.4f} ± {placement_data['B_lambda'].std():.4f}")
    
    # 6. Resolution effects
    print(f"\n6. MESH RESOLUTION EFFECTS:")
    resolution_effects = df.groupby('n_x').agg({
        'B_lambda': ['mean', 'std'],
        'eta_h': 'mean'
    }).round(4)
    
    for nx in sorted(df['n_x'].unique()):
        nx_data = df[df['n_x'] == nx]
        print(f"   • n_x = {nx}: B_λ = {nx_data['B_lambda'].mean():.4f}, η_h = {nx_data['eta_h'].mean():.6f}")
    
    # 7. MCMC convergence analysis
    print(f"\n7. MCMC CONVERGENCE:")
    convergence_rate = df['converged'].mean()
    print(f"   • Overall convergence rate: {convergence_rate:.3f}")
    print(f"   • Mean acceptance rate: {df['acceptance_rate'].mean():.4f}")
    
    conv_by_lambda = df.groupby('lambda')['converged'].mean()
    print("   Convergence by temperature:")
    for lam in sorted(df['lambda'].unique()):
        rate = conv_by_lambda[lam]
        print(f"   • λ = {lam}: {rate:.3f} convergence rate")
    
    # 8. Certificate gap analysis
    print(f"\n8. CERTIFICATE GAP ANALYSIS:")
    df['cert_gap'] = df['B_lambda'] - df['L_mc']
    gap_stats = df['cert_gap'].describe()
    print(f"   • Mean certificate gap (B_λ - L_MC): {gap_stats['mean']:.4f}")
    print(f"   • Median certificate gap: {gap_stats['50%']:.4f}")
    print(f"   • Gap range: [{gap_stats['min']:.4f}, {gap_stats['max']:.4f}]")
    
    # 9. Best configurations
    print(f"\n9. BEST PERFORMING CONFIGURATIONS:")
    best_configs = df.nsmallest(5, 'B_lambda')[['s', 'placement_type', 'sigma', 'n_x', 'lambda', 'm', 'B_lambda', 'L_mc']]
    print("   Top 5 configurations (lowest B_λ):")
    for i, (_, row) in enumerate(best_configs.iterrows(), 1):
        print(f"   {i}. s={int(row['s'])}, {row['placement_type']}, σ={row['sigma']}, n_x={int(row['n_x'])}, λ={row['lambda']}, m={int(row['m'])}")
        print(f"      → B_λ = {row['B_lambda']:.4f}, L_MC = {row['L_mc']:.4f}")
    
    # 10. Summary insights
    print(f"\n" + "=" * 60)
    print("🎯 KEY INSIGHTS SUMMARY")
    print("=" * 60)
    
    print(f"✅ Certificate Reliability: {validity_rate*100:.1f}% of certificates are valid")
    print(f"🌡️ Optimal Temperature: λ = {best_lambda} provides tightest bounds")
    print(f"📡 Sensor Impact: More sensors ({max(df['s'])}) reduce certificate by {(sensor_effects.loc[min(df['s'])]['mean'] - sensor_effects.loc[max(df['s'])]['mean']):.3f}")
    print(f"🔧 Resolution Benefit: Higher resolution (n_x=100) improves discretization penalty")
    print(f"📊 MCMC Performance: {convergence_rate*100:.1f}% convergence rate with {df['acceptance_rate'].mean():.1%} acceptance")
    
    # Save analysis summary
    output_path = Path('results_full_section_a/analysis_summary.txt')
    with open(output_path, 'w') as f:
        f.write("PAC-Bayes Inverse PDE - Section A Results Analysis\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Certificate validity rate: {validity_rate:.3f}\n")
        f.write(f"Best temperature parameter: λ = {best_lambda}\n")
        f.write(f"Convergence rate: {convergence_rate:.3f}\n")
        f.write(f"Mean certificate gap: {df['cert_gap'].mean():.4f}\n")
    
    print(f"\n📄 Analysis summary saved to: {output_path}")

if __name__ == '__main__':
    main()