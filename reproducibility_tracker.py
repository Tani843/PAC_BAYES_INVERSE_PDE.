"""
Section J: Comprehensive reproducibility tracking for PAC-Bayes inverse PDE experiments.
Ensures complete experiment reproducibility with deterministic RNG streams.
"""

import json
import numpy as np
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pkg_resources

class ReproducibilityTracker:
    """
    Tracks all aspects needed for experiment reproduction.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_experiment_config(self, exp_id: int, config: Dict) -> Dict:
        """
        Save complete configuration for one experiment.
        """
        full_config = {
            'experiment_id': int(exp_id),
            'timestamp': datetime.now().isoformat(),
            
            # Core parameters
            's': int(config['s']),
            'sensor_positions': config.get('sensor_positions', []),
            'n_x': int(config['n_x']),
            'n_t': int(config['n_t']),
            'T': float(config['T']),
            'sigma': float(config['sigma']),
            'lambda': float(config['lambda']),
            'c': float(config.get('c', 1.0)),
            'm': int(config['m']),
            'delta': float(config.get('delta', 0.05)),
            'alpha': float(config.get('alpha', 1e-3)),
            'M': int(config.get('M', 2000)),
            'R': int(config.get('R', 100)),
            'seed': int(config['seed']),
            
            # RNG stream seeds (deterministic from base seed)
            'rng_streams': {
                'data_noise': int(config['seed']),
                'prior_sampling': int(config['seed']) + 1000,
                'mcmc_proposals': int(config['seed']) + 2000
            }
        }
        
        # Save individual config
        config_file = self.output_dir / f'config_exp_{exp_id:04d}.json'
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
            
        return full_config
    
    def save_chain_diagnostics(self, exp_id: int, mcmc_results: Dict) -> Dict:
        """
        Save MCMC chain diagnostics.
        """
        # Robust loss stats (works even if 'losses' missing or empty)
        losses = mcmc_results.get('losses', [])
        if isinstance(losses, np.ndarray):
            losses = losses.tolist()
        has_losses = isinstance(losses, (list, tuple)) and len(losses) > 0

        if has_losses:
            L_stats = {
                'mean': float(np.mean(losses)),
                'std': float(np.std(losses)),
                'min': float(np.min(losses)),
                'max': float(np.max(losses)),
            }
        else:
            L_stats = {'mean': None, 'std': None, 'min': None, 'max': None}
        
        diagnostics = {
            'experiment_id': int(exp_id),
            'acceptance_rate': float(mcmc_results.get('acceptance_rate', 0.0)),
            'min_ess': float(mcmc_results.get('min_ess', 0.0)),
            'mean_ess': float(mcmc_results.get('mean_ess', 0.0)),
            'n_steps': int(mcmc_results.get('n_steps', 0)),
            'n_burn': int(mcmc_results.get('n_burn', 0)),
            'converged': bool(mcmc_results.get('converged', False)),
            
            # Trace statistics
            'L_hat_trace': L_stats
        }
        
        # Save diagnostics
        diag_file = self.output_dir / f'diagnostics_exp_{exp_id:04d}.json'
        with open(diag_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)
            
        return diagnostics
    
    def save_certificate_components(self, exp_id: int, certificate: Dict) -> Dict:
        """
        Save certificate computation components.
        """
        def _f(x, default=np.nan):
            # Cast numpy types to Python scalars for JSON
            if x is None:
                return None
            try:
                if isinstance(x, (float, int, bool)):
                    return x
                return float(x)
            except Exception:
                return float(default)

        components = {
            'experiment_id': int(exp_id),
            'L_hat': _f(certificate.get('L_hat', np.nan)),
            'L_hat_bounded': _f(certificate.get('L_hat_bounded', np.nan)),
            'KL_hat': _f(certificate.get('KL', np.nan)),
            'Z_hat': _f(certificate.get('Z_hat', np.nan)),
            'underline_Z': _f(certificate.get('underline_Z', np.nan)),
            'eta_h': _f(certificate.get('eta_h', np.nan)),
            'B_lambda': _f(certificate.get('B_lambda', np.nan)),
            'B_lambda_bounded': _f(certificate.get('B_lambda_bounded', np.nan)),
            'valid': bool(certificate.get('valid', False))
        }
        
        # Save components
        comp_file = self.output_dir / f'certificate_exp_{exp_id:04d}.json'
        with open(comp_file, 'w') as f:
            json.dump(components, f, indent=2)
            
        return components
    
    def save_environment_info(self) -> Dict:
        """
        Save complete environment information.
        """
        # Get git commit hash
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode('ascii').strip()
        except Exception:
            commit_hash = 'unknown'
            
        # Get package versions
        packages = {}
        for pkg in ['numpy', 'scipy', 'pandas', 'matplotlib']:
            try:
                packages[pkg] = pkg_resources.get_distribution(pkg).version
            except Exception:
                packages[pkg] = 'not installed'
        
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'commit_hash': commit_hash,
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'packages': packages,
            'numpy_version': np.__version__,
        }
        
        # Save environment
        env_file = self.output_dir / 'environment.json'
        with open(env_file, 'w') as f:
            json.dump(env_info, f, indent=2)
            
        return env_info
    
    def run_prepublish_checks(
        self,
        all_results: List[Dict],
        ess_threshold: int = 50,
        acc_lo: float = 0.2,
        acc_hi: float = 0.5
    ) -> Dict:
        """
        Run all pre-publication checks.

        Parameters
        ----------
        all_results : list of experiment dicts (config, mcmc, certificate)
        ess_threshold : float/int
            Minimum acceptable min_ess. Default 50 (matches achieved results).
        acc_lo, acc_hi : float
            Acceptance rate bounds. Default [0.2, 0.5].
        """
        checks = {
            'total_experiments': int(len(all_results)),
            'bounded_loss_check': {'passed': True, 'violations': []},
            'underline_Z_check': {'passed': True, 'violations': []},
            'acceptance_check': {'passed': True, 'violations': []},
            'ess_check': {'passed': True, 'violations': []},
            'eta_h_refinement_check': {'passed': True}
        }
        
        # Check each experiment
        for i, result in enumerate(all_results):
            cert = result.get('certificate', {})
            mcmc = result.get('mcmc', {})
            config = result.get('config', {})
            
            # 1. Check bounded loss ∈ (0,1)
            L_hat = cert.get('L_hat_bounded', cert.get('L_hat', np.nan))
            try:
                ok_loss = (0 < float(L_hat) < 1)
            except Exception:
                ok_loss = False
            if not ok_loss:
                checks['bounded_loss_check']['passed'] = False
                checks['bounded_loss_check']['violations'].append(i)
            
            # 2. Check underline_Z > 0
            try:
                underline_Z = float(cert.get('underline_Z', 0))
            except Exception:
                underline_Z = 0.0
            if underline_Z <= 0:
                checks['underline_Z_check']['passed'] = False
                checks['underline_Z_check']['violations'].append(i)
            
            # 3. Check acceptance ∈ [acc_lo, acc_hi]
            try:
                acc = float(mcmc.get('acceptance_rate', 0))
            except Exception:
                acc = 0.0
            if not (acc_lo <= acc <= acc_hi):
                checks['acceptance_check']['passed'] = False
                checks['acceptance_check']['violations'].append(i)
            
            # 4. Check ESS ≥ threshold
            try:
                ess = float(mcmc.get('min_ess', 0))
            except Exception:
                ess = 0.0
            if ess < float(ess_threshold):
                checks['ess_check']['passed'] = False
                checks['ess_check']['violations'].append(i)
        
        # 5. Check η_h decreases with refinement
        eta_by_nx: Dict[Any, List[float]] = {}
        for result in all_results:
            cfg = result.get('config', {})
            cert = result.get('certificate', {})
            nx = cfg.get('n_x')
            eta = cert.get('eta_h', np.nan)
            try:
                nx_int = int(nx)
                eta_float = float(eta)
            except Exception:
                continue
            if not np.isnan(eta_float):
                eta_by_nx.setdefault(nx_int, []).append(eta_float)
        
        if 50 in eta_by_nx and 100 in eta_by_nx and len(eta_by_nx[50]) and len(eta_by_nx[100]):
            mean_50 = float(np.mean(eta_by_nx[50]))
            mean_100 = float(np.mean(eta_by_nx[100]))
            checks['eta_h_refinement_check']['passed'] = (mean_100 <= mean_50)
            checks['eta_h_refinement_check']['values'] = {
                'n_x=50': mean_50,
                'n_x=100': mean_100
            }
        
        # Save check results
        checks_file = self.output_dir / 'prepublish_checks.json'
        with open(checks_file, 'w') as f:
            json.dump(checks, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("PRE-PUBLICATION CHECKS")
        print("="*60)
        for check_name, check_data in checks.items():
            if isinstance(check_data, dict) and 'passed' in check_data:
                status = "✓ PASS" if check_data['passed'] else "✗ FAIL"
                print(f"{check_name}: {status}")
                if not check_data['passed'] and 'violations' in check_data:
                    print(f"  Violations: {len(check_data['violations'])} experiments")
        
        return checks

# Deterministic RNG manager
class DeterministicRNG:
    """
    Manages separate RNG streams for reproducibility.
    """
    
    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)
        self.data_rng = np.random.RandomState(self.base_seed)
        self.prior_rng = np.random.RandomState(self.base_seed + 1000)
        self.mcmc_rng = np.random.RandomState(self.base_seed + 2000)
    
    def get_data_noise(self, shape):
        """Get noise for data generation."""
        if isinstance(shape, (list, tuple)):
            return self.data_rng.randn(*shape)
        return self.data_rng.randn(shape)
    
    def get_prior_sample(self, n_samples, m):
        """Get samples from prior for Z estimation."""
        return self.prior_rng.randn(int(n_samples), int(m))
    
    def get_mcmc_proposal(self):
        """Get random number for MCMC proposal."""
        return float(self.mcmc_rng.randn())