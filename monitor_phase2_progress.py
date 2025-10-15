#!/usr/bin/env python3
"""
Monitor Phase 2 full grid execution progress
"""

import json
import time
from pathlib import Path
from datetime import datetime
import sys

def find_latest_results_dir():
    """Find the most recent Phase 2 results directory."""
    results_dirs = list(Path('.').glob('results_phase2_full_*'))
    if not results_dirs:
        return None
    return max(results_dirs, key=lambda p: p.stat().st_mtime)

def parse_checkpoint(checkpoint_file):
    """Parse checkpoint file to get progress statistics."""
    try:
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
        
        total = len(results)
        successful = [r for r in results if 'error' not in r]
        converged = sum(1 for r in successful if r['mcmc']['converged'])
        valid_certs = sum(1 for r in successful if r['certificate']['valid'])
        
        if successful:
            acceptance_rates = [r['mcmc']['acceptance_rate'] for r in successful]
            ess_values = [r['mcmc']['ess_min'] for r in successful]
            runtimes = [r['performance']['runtime'] for r in successful]
            
            return {
                'total': total,
                'successful': len(successful),
                'errors': total - len(successful),
                'converged': converged,
                'valid_certificates': valid_certs,
                'convergence_rate': converged / len(successful) if successful else 0,
                'validity_rate': valid_certs / len(successful) if successful else 0,
                'mean_acceptance': sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0,
                'mean_ess': sum(ess_values) / len(ess_values) if ess_values else 0,
                'mean_runtime': sum(runtimes) / len(runtimes) if runtimes else 0,
                'total_runtime': sum(runtimes) if runtimes else 0
            }
    except Exception as e:
        return {'error': str(e)}
    
    return None

def monitor_progress():
    """Monitor the Phase 2 execution progress."""
    
    print("=" * 70)
    print("MONITORING PHASE 2 FULL GRID EXECUTION")
    print("=" * 70)
    
    results_dir = find_latest_results_dir()
    if not results_dir:
        print("❌ No Phase 2 results directory found")
        print("   Make sure run_full_grid_phase2.py is running")
        return False
    
    print(f"📁 Monitoring directory: {results_dir}")
    print(f"⏰ Started monitoring at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    last_checkpoint = None
    start_time = time.time()
    
    try:
        while True:
            # Find latest checkpoint
            checkpoints = sorted(results_dir.glob('checkpoint_*.json'))
            
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                
                # Only update if new checkpoint
                if latest_checkpoint != last_checkpoint:
                    stats = parse_checkpoint(latest_checkpoint)
                    
                    if stats and 'error' not in stats:
                        elapsed = time.time() - start_time
                        remaining_experiments = 1728 - stats['total']
                        
                        if stats['mean_runtime'] > 0:
                            eta_seconds = remaining_experiments * stats['mean_runtime']
                            eta_hours = eta_seconds / 3600
                        else:
                            eta_hours = 0
                        
                        print(f"🔄 Progress Update - {datetime.now().strftime('%H:%M:%S')}")
                        print(f"   Experiments: {stats['total']}/1728 ({stats['total']/1728:.1%})")
                        print(f"   Success rate: {stats['successful']}/{stats['total']} ({stats['successful']/stats['total']:.1%})")
                        print(f"   Convergence: {stats['converged']}/{stats['successful']} ({stats['convergence_rate']:.1%})")
                        print(f"   Valid certs: {stats['valid_certificates']}/{stats['successful']} ({stats['validity_rate']:.1%})")
                        print(f"   Performance: acc={stats['mean_acceptance']:.3f}, ESS={stats['mean_ess']:.1f}")
                        print(f"   Timing: {stats['mean_runtime']:.1f}s/exp, total={stats['total_runtime']/3600:.1f}h")
                        print(f"   ETA: {eta_hours:.1f} hours remaining")
                        print()
                        
                        last_checkpoint = latest_checkpoint
                
            # Check if execution is complete
            final_results = results_dir / 'section_a_phase2_complete.json'
            if final_results.exists():
                print("🎉 EXECUTION COMPLETE!")
                
                try:
                    with open(final_results, 'r') as f:
                        final_data = json.load(f)
                    
                    # Final statistics
                    successful = [r for r in final_data if 'error' not in r]
                    converged = sum(1 for r in successful if r['mcmc']['converged'])
                    valid_certs = sum(1 for r in successful if r['certificate']['valid'])
                    
                    print(f"📊 FINAL RESULTS:")
                    print(f"   Total: {len(final_data)}/1728 experiments")
                    print(f"   Success: {len(successful)}/{len(final_data)} ({len(successful)/len(final_data):.1%})")
                    print(f"   Convergence: {converged}/{len(successful)} ({converged/len(successful):.1%})")
                    print(f"   Valid certificates: {valid_certs}/{len(successful)} ({valid_certs/len(successful):.1%})")
                    
                    # Compare with original
                    original_convergence = 0.0
                    original_validity = 0.014
                    
                    conv_improvement = (converged/len(successful)) / max(0.001, original_convergence)
                    validity_improvement = (valid_certs/len(successful)) / original_validity
                    
                    print(f"🚀 IMPROVEMENTS:")
                    print(f"   Convergence: {conv_improvement:.0f}x better than original")
                    print(f"   Validity: {validity_improvement:.1f}x better than original")
                    
                except Exception as e:
                    print(f"   (Error reading final results: {e})")
                
                break
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoring stopped by user")
        return True
    
    print(f"\n✅ Monitoring complete!")
    return True

def show_current_status():
    """Show current status without continuous monitoring."""
    
    results_dir = find_latest_results_dir()
    if not results_dir:
        print("❌ No Phase 2 results directory found")
        return
    
    print(f"📁 Results directory: {results_dir}")
    
    # Check for final results first
    final_results = results_dir / 'section_a_phase2_complete.json'
    if final_results.exists():
        print("✅ Execution is COMPLETE!")
        try:
            with open(final_results, 'r') as f:
                data = json.load(f)
            print(f"   Final count: {len(data)} experiments")
        except:
            print("   (Could not read final results)")
        return
    
    # Find latest checkpoint
    checkpoints = sorted(results_dir.glob('checkpoint_*.json'))
    if checkpoints:
        latest = checkpoints[-1]
        stats = parse_checkpoint(latest)
        
        if stats and 'error' not in stats:
            print(f"📊 Latest Progress ({latest.name}):")
            print(f"   Experiments: {stats['total']}/1728 ({stats['total']/1728:.1%})")
            print(f"   Success rate: {stats['successful']}/{stats['total']} ({stats['successful']/stats['total']:.1%})")
            print(f"   Convergence: {stats['convergence_rate']:.1%}")
            print(f"   Valid certificates: {stats['validity_rate']:.1%}")
            print(f"   Avg acceptance: {stats['mean_acceptance']:.3f}")
            print(f"   Avg ESS: {stats['mean_ess']:.1f}")
            print(f"   Runtime so far: {stats['total_runtime']/3600:.1f} hours")
        else:
            print("❌ Could not parse latest checkpoint")
    else:
        print("⏳ Execution may be starting (no checkpoints yet)")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        show_current_status()
    else:
        monitor_progress()