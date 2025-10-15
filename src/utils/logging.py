"""
Section J: Logging Utilities
Comprehensive logging for diagnostics and reproducibility checks
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class DiagnosticsLogger:
    """
    Logs detailed diagnostics for each experiment component.
    """
    
    def __init__(self, output_dir: str = 'results/logs'):
        """
        Initialize diagnostics logger.
        
        Args:
            output_dir: Directory for diagnostic logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for diagnostics
        self.chain_diagnostics = []
        self.loss_diagnostics = []
        self.certificate_diagnostics = []
        self.convergence_diagnostics = []
    
    def log_chain_diagnostics(self, 
                            config: Dict,
                            chain: np.ndarray,
                            acceptance_rate: float,
                            ess: np.ndarray,
                            acf: np.ndarray):
        """
        Log MCMC chain diagnostics as specified in Section E.
        
        Args:
            config: Experiment configuration
            chain: MCMC samples
            acceptance_rate: Acceptance rate
            ess: Effective sample size per coordinate
            acf: Autocorrelation function
        """
        diagnostic = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                's': config['s'],
                'sigma': config['sigma'],
                'lambda': config.get('lambda'),
                'seed': config['seed']
            },
            'n_steps': len(chain),
            'n_params': chain.shape[1] if len(chain.shape) > 1 else 1,
            'acceptance_rate': float(acceptance_rate),
            'ess_per_coord': ess.tolist() if isinstance(ess, np.ndarray) else ess,
            'min_ess': float(np.min(ess)),
            'mean_ess': float(np.mean(ess)),
            'acceptance_in_range': 0.2 <= acceptance_rate <= 0.5,
            'ess_sufficient': bool(np.all(ess >= 500))
        }
        
        self.chain_diagnostics.append(diagnostic)
        
        # Save detailed chain trace
        chain_file = self.output_dir / f"chain_s{config['s']}_sigma{config['sigma']}_seed{config['seed']}.npz"
        np.savez_compressed(
            chain_file,
            chain=chain,
            ess=ess,
            acf=acf,
            config=config
        )
        
        # Log warning if convergence criteria not met
        if not diagnostic['acceptance_in_range']:
            print(f"? Acceptance rate {acceptance_rate:.3f} outside [0.2, 0.5]")
        if not diagnostic['ess_sufficient']:
            print(f"? Minimum ESS {np.min(ess):.1f} < 500")
    
    def log_loss_components(self,
                           config: Dict,
                           empirical_loss: float,
                           bounded_loss: float,
                           squared_loss: float):
        """
        Log loss function evaluations for verification.
        
        Args:
            config: Experiment configuration
            empirical_loss: Empirical bounded loss
            bounded_loss: Bounded loss value
            squared_loss: Standard squared loss
        """
        diagnostic = {
            'timestamp': datetime.now().isoformat(),
            'config_hash': self._config_hash(config),
            'empirical_loss': float(empirical_loss),
            'bounded_loss': float(bounded_loss),
            'squared_loss': float(squared_loss),
            'loss_in_bounds': 0 < bounded_loss < 1
        }
        
        self.loss_diagnostics.append(diagnostic)
        
        # Check boundedness
        if not diagnostic['loss_in_bounds']:
            print(f"? Loss {bounded_loss:.4f} not in (0,1)")
    
    def log_certificate_components(self,
                                  config: Dict,
                                  L_hat: float,
                                  KL: float,
                                  Z_hat: float,
                                  underline_Z: float,
                                  eta_h: float,
                                  B_lambda: float):
        """
        Log PAC-Bayes certificate components for verification.
        
        Args:
            config: Experiment configuration
            L_hat: Empirical loss
            KL: KL divergence
            Z_hat: Partition function estimate
            underline_Z: Conservative Z estimate
            eta_h: Discretization penalty
            B_lambda: Final certificate
        """
        diagnostic = {
            'timestamp': datetime.now().isoformat(),
            's': config['s'],
            'sigma': config['sigma'],
            'lambda': config['lambda'],
            'n': config['n'],
            'delta': config['delta'],
            'L_hat': float(L_hat),
            'KL': float(KL),
            'Z_hat': float(Z_hat),
            'underline_Z': float(underline_Z),
            'underline_Z_positive': underline_Z > 0,
            'eta_h': float(eta_h),
            'B_lambda': float(B_lambda),
            'certificate_valid': B_lambda >= L_hat
        }
        
        self.certificate_diagnostics.append(diagnostic)
        
        # Pre-publish checks as specified
        if not diagnostic['underline_Z_positive']:
            print(f"? underline_Z = {underline_Z:.6f} <= 0, increase M")
        if not diagnostic['certificate_valid']:
            print(f"? Certificate violated: B_? = {B_lambda:.4f} < L? = {L_hat:.4f}")
    
    def log_refinement_test(self,
                           n_x_coarse: int,
                           n_x_fine: int,
                           eta_coarse: float,
                           eta_fine: float):
        """
        Log mesh refinement test results.
        
        Args:
            n_x_coarse: Coarse mesh size
            n_x_fine: Fine mesh size  
            eta_coarse: Discretization error on coarse mesh
            eta_fine: Discretization error on fine mesh
        """
        diagnostic = {
            'timestamp': datetime.now().isoformat(),
            'n_x_coarse': n_x_coarse,
            'n_x_fine': n_x_fine,
            'eta_coarse': float(eta_coarse),
            'eta_fine': float(eta_fine),
            'eta_decreases': eta_fine < eta_coarse,
            'reduction_factor': float(eta_coarse / eta_fine) if eta_fine > 0 else np.inf
        }
        
        self.convergence_diagnostics.append(diagnostic)
        
        if not diagnostic['eta_decreases']:
            print(f"? ?_h did not decrease on refinement: {eta_coarse:.4f} ? {eta_fine:.4f}")
    
    def save_all_diagnostics(self, experiment_id: str):
        """
        Save all diagnostics to files.
        
        Args:
            experiment_id: Unique experiment identifier
        """
        # Save as JSON
        all_diagnostics = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'chain_diagnostics': self.chain_diagnostics,
            'loss_diagnostics': self.loss_diagnostics,
            'certificate_diagnostics': self.certificate_diagnostics,
            'convergence_diagnostics': self.convergence_diagnostics
        }
        
        diagnostics_file = self.output_dir / f"{experiment_id}_diagnostics.json"
        with open(diagnostics_file, 'w') as f:
            json.dump(all_diagnostics, f, indent=2)
        
        # Create summary DataFrame for certificates
        if self.certificate_diagnostics:
            df = pd.DataFrame(self.certificate_diagnostics)
            summary_file = self.output_dir / f"{experiment_id}_certificate_summary.csv"
            df.to_csv(summary_file, index=False)
            
            # Print summary statistics
            print("\nCertificate Summary:")
            print(f"  Valid certificates: {df['certificate_valid'].sum()}/{len(df)}")
            print(f"  Mean B_?: {df['B_lambda'].mean():.4f}")
            print(f"  Mean ?_h: {df['eta_h'].mean():.6f}")
    
    def _config_hash(self, config: Dict) -> str:
        """Generate hash of configuration for tracking."""
        import hashlib
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class PerformanceTracker:
    """
    Track computational performance metrics.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.timings = {}
        self.memory_usage = {}
        self.solver_calls = 0
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing a section."""
        import time
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and record duration."""
        import time
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            del self.start_times[name]
            return duration
        return 0.0
    
    def record_memory(self, name: str):
        """Record current memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if name not in self.memory_usage:
            self.memory_usage[name] = []
        self.memory_usage[name].append(memory_mb)
    
    def increment_solver_calls(self):
        """Increment solver call counter."""
        self.solver_calls += 1
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        summary = {
            'total_solver_calls': self.solver_calls,
            'timings': {},
            'memory_peak_mb': {}
        }
        
        for name, times in self.timings.items():
            summary['timings'][name] = {
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'count': len(times)
            }
        
        for name, memory in self.memory_usage.items():
            summary['memory_peak_mb'][name] = max(memory)
        
        return summary
    
    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        
        print("\nPerformance Summary:")
        print("=" * 50)
        print(f"Total solver calls: {summary['total_solver_calls']}")
        
        print("\nTimings:")
        for name, stats in summary['timings'].items():
            print(f"  {name}:")
            print(f"    Total: {stats['total']:.2f}s")
            print(f"    Mean: {stats['mean']:.3f}s")
            print(f"    Count: {stats['count']}")
        
        if summary['memory_peak_mb']:
            print("\nPeak Memory Usage:")
            for name, peak in summary['memory_peak_mb'].items():
                print(f"  {name}: {peak:.1f} MB")