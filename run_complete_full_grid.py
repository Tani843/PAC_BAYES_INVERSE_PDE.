#!/usr/bin/env python3
"""
Run the complete 1,728 experiment Section A grid using the fixed configuration
This demonstrates the full parameter space coverage
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Import the runner but use the now-fixed original config
from run_full_section_a_grid import run_full_section_a_grid

def main():
    """Run the complete Section A experiment grid."""
    
    print("=" * 80)
    print("LAUNCHING COMPLETE SECTION A EXPERIMENT GRID")
    print("=" * 80)
    print("Configuration: Fixed experiment_config.py")
    print("Total experiments: 1,728")
    print("Grid structure: 2×2×3×2×2×3×1×2×2×3")
    print("Estimated runtime: ~6 minutes")
    print("=" * 80)
    
    try:
        results = run_full_section_a_grid()
        
        print(f"\n🎉 SUCCESS! {len(results)} experiments completed!")
        print(f"Results saved in: results_full_section_a/")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Experiment grid interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Error running full grid: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Starting complete Section A experiment grid execution...")
    print("Press Ctrl+C to interrupt if needed.")
    print()
    
    success = main()
    
    if success:
        print("\n✅ Complete Section A grid execution finished successfully!")
    else:
        print("\n❌ Grid execution did not complete successfully.")