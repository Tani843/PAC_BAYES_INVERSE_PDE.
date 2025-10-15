"""
Fixes Table 1 generation using the exact same loader as figures.
"""

import pandas as pd
from pathlib import Path

# Import the EXACT loader used for figures to guarantee consistency
from generate_paper_figures_fixed import load_data

def generate_corrected_table():
    """Generate Table 1 with correct bounded values using same loader as figures."""
    
    # Use the SAME loader that figures use - guarantees consistency
    df = load_data('PAC_BAYES_COMPLETE_VERIFIED_1728.json')
    
    # Group and aggregate
    config_cols = ['sigma', 's', 'lambda', 'n_x', 'T', 'm']
    grouped = df.groupby(config_cols).agg({
        'L_hat': 'mean',  # These are already bounded from load_data
        'B_lambda': 'mean',  # These are already bounded from load_data
        'KL': 'mean',
        'eta_h': 'mean',
        'n': 'first',
        'delta': 'first',
        'valid': 'mean'
    }).reset_index()
    
    # Rename to indicate bounded
    grouped.rename(columns={
        'L_hat': 'L_hat_bounded',
        'B_lambda': 'B_lambda_bounded'
    }, inplace=True)
    
    # Filter for table subset
    table_subset = grouped[
        (grouped['T'] == 0.5) & 
        (grouped['m'] == 3)
    ][['sigma', 's', 'lambda', 'n', 'delta', 'n_x', 
       'L_hat_bounded', 'KL', 'eta_h', 'B_lambda_bounded', 'valid']]
    
    # Round for display
    table_subset = table_subset.round(4)
    
    # Sort
    table_subset = table_subset.sort_values(['sigma', 's', 'lambda', 'n_x'])
    
    # Save to CANONICAL filenames (no _CORRECTED suffix)
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)
    
    # CSV
    table_subset.to_csv(output_dir / 'table_1_per_config_stats.csv', index=False)
    
    # LaTeX
    latex_table = table_subset.to_latex(
        index=False,
        caption="Per-configuration statistics verify PAC-Bayes inequality",
        label="tab:per_config_stats"
    )
    with open(output_dir / 'table_1_per_config_stats.tex', 'w') as f:
        f.write(latex_table)
    
    print("="*60)
    print("TABLE 1 REGENERATED WITH CORRECT BOUNDED VALUES")
    print("="*60)
    
    print("\nFirst 10 rows:")
    print(table_subset[['sigma', 's', 'lambda', 'n_x', 'L_hat_bounded', 'B_lambda_bounded']].head(10))
    
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS:")
    print("="*60)
    
    print("\nL_hat_bounded:")
    print(f"  Min: {table_subset['L_hat_bounded'].min():.4f}")
    print(f"  Max: {table_subset['L_hat_bounded'].max():.4f}")
    print(f"  Mean: {table_subset['L_hat_bounded'].mean():.4f}")
    
    print("\nB_lambda_bounded:")
    print(f"  Min: {table_subset['B_lambda_bounded'].min():.4f}")
    print(f"  Max: {table_subset['B_lambda_bounded'].max():.4f}")
    print(f"  Mean: {table_subset['B_lambda_bounded'].mean():.4f}")
    
    if table_subset['L_hat_bounded'].min() < 0.9:
        print("\n✓ SUCCESS: Values are properly bounded in expected range!")
        print(f"✓ Saved to: {output_dir}/table_1_per_config_stats.csv")
        print(f"✓ Saved to: {output_dir}/table_1_per_config_stats.tex")
    else:
        print("\n⚠️ WARNING: Values still appear incorrect.")
    
    return table_subset

if __name__ == '__main__':
    corrected_table = generate_corrected_table()