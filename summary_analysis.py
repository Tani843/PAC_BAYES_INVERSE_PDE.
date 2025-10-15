# summary_analysis.py
import json
import pandas as pd
import numpy as np

with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
    data = json.load(f)

# Key metrics
success_rate = sum(1 for exp in data if exp.get('status') == 'success') / len(data)
valid_certs = sum(1 for exp in data 
                  if exp.get('status') == 'success' 
                  and exp.get('certificate', {}).get('B_lambda', 0) >= 
                      exp.get('certificate', {}).get('L_hat', 1)) / len(data)

print(f"Overall success rate: {success_rate:.1%}")
print(f"Valid certificates: {valid_certs:.1%}")