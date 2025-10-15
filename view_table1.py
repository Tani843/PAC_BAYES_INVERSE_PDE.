# view_table1.py
import pandas as pd

# Load and display the full table
table1 = pd.read_csv('table1_per_config_results.csv')
print("Table 1: Per-Config Results (L̂, KL, η_h, B_λ)")
print("="*80)
print(table1.to_string())

# Also save as LaTeX for paper
with open('table1_latex.tex', 'w') as f:
    f.write(table1.to_latex(
        caption="Per-configuration PAC-Bayes certificate components",
        label="tab:results",
        float_format="%.4f"
    ))
print("\nLaTeX version saved to table1_latex.tex")