#!/usr/bin/env python3
"""
Diagnose status breakdown of complete PAC-Bayes experiment dataset
"""

import json
from pathlib import Path
from collections import Counter

def diagnose_experiment_status():
    """Analyze the status of all experiments in the complete dataset"""
    
    print("=" * 60)
    print("DIAGNOSING EXPERIMENT STATUS BREAKDOWN")
    print("=" * 60)
    
    # Load your complete dataset
    print("Loading complete dataset...")
    with open('PAC_BAYES_COMPLETE_VERIFIED_1728.json', 'r') as f:
        data = json.load(f)

    print(f"Total experiments loaded: {len(data)}")

    # Count statuses
    statuses = Counter()
    for exp in data:
        status = exp.get('status', 'no_status')
        statuses[status] += 1

    print("\n📊 STATUS BREAKDOWN:")
    for status, count in statuses.items():
        pct = (count/len(data))*100
        symbol = "✅" if status == 'success' else "⚠️" if status == 'timeout' else "❌"
        print(f"  {symbol} {status}: {count} ({pct:.1f}%)")

    # Check which experiments aren't 'success'
    non_success = []
    non_success_details = []
    
    for i, exp in enumerate(data):
        if exp.get('status') != 'success':
            non_success.append(i)
            # Get experiment details
            config = exp.get('config', {})
            exp_detail = {
                'exp_num': i,
                'status': exp.get('status'),
                's': config.get('s'),
                'sigma': config.get('sigma'),
                'lambda': config.get('lambda'),
                'T': config.get('T'),
                'm': config.get('m'),
                'seed': config.get('seed'),
                'error': exp.get('error', 'N/A')[:100] + '...' if exp.get('error', '') else 'N/A'
            }
            non_success_details.append(exp_detail)

    print(f"\n🔍 NON-SUCCESS EXPERIMENTS ANALYSIS:")
    if non_success:
        print(f"Total non-success: {len(non_success)}")
        print(f"Experiment numbers: {non_success[:20]}{'...' if len(non_success) > 20 else ''}")
        
        print(f"\n📋 DETAILED BREAKDOWN (first 10):")
        for detail in non_success_details[:10]:
            print(f"  exp_{detail['exp_num']:04d}: {detail['status']} - "
                  f"s={detail['s']}, σ={detail['sigma']}, λ={detail['lambda']}, "
                  f"T={detail['T']}, m={detail['m']}, seed={detail['seed']}")
            if detail['error'] != 'N/A':
                print(f"    Error: {detail['error']}")
        
        # Analyze patterns in failures
        print(f"\n🔍 FAILURE PATTERN ANALYSIS:")
        failure_params = {
            's': Counter(),
            'sigma': Counter(),
            'lambda': Counter(),
            'T': Counter(),
            'm': Counter(),
            'seed': Counter(),
            'status': Counter()
        }
        
        for detail in non_success_details:
            for param in ['s', 'sigma', 'lambda', 'T', 'm', 'seed', 'status']:
                if detail[param] is not None:
                    failure_params[param][detail[param]] += 1
        
        for param, counts in failure_params.items():
            if counts:
                print(f"  {param}: {dict(counts)}")
                
    else:
        print("  🎉 No non-success experiments found!")

    # Runtime analysis for successful experiments
    print(f"\n⏱️ RUNTIME ANALYSIS (SUCCESS ONLY):")
    successful_runtimes = []
    for exp in data:
        if exp.get('status') == 'success' and 'performance' in exp:
            runtime = exp['performance'].get('runtime')
            if runtime and runtime > 0:
                successful_runtimes.append(runtime)
    
    if successful_runtimes:
        import numpy as np
        print(f"  Successful experiments with runtime: {len(successful_runtimes)}")
        print(f"  Average runtime: {np.mean(successful_runtimes):.1f}s")
        print(f"  Median runtime: {np.median(successful_runtimes):.1f}s")
        print(f"  Min runtime: {np.min(successful_runtimes):.1f}s")
        print(f"  Max runtime: {np.max(successful_runtimes):.1f}s")
        print(f"  Total compute time: {sum(successful_runtimes)/3600:.1f} hours")

    print("\n" + "=" * 60)
    success_rate = (statuses['success'] / len(data)) * 100
    if success_rate >= 95:
        print(f"🎉 EXCELLENT SUCCESS RATE: {success_rate:.1f}%")
        print("🎉 Dataset ready for comprehensive analysis!")
    else:
        print(f"⚠️ SUCCESS RATE: {success_rate:.1f}%")
        print("ℹ️ Consider investigating failure patterns")

if __name__ == '__main__':
    diagnose_experiment_status()