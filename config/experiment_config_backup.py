"""
Section A: Global Configuration (final)
Experiment grid and global parameters for PAC-Bayes certification experiments
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import itertools

@dataclass
class ExperimentConfig:
    """
    Global configuration for PAC-Bayes inverse PDE experiments.
    Implements Section A of the specification document.
    """
    
    # Sensor configurations
    s_values: List[int] = field(default_factory=lambda: [3, 5])
    
    # Sensor placements (exactly as specified)
    sensor_placements: Dict[int, Dict[str, List[float]]] = field(default_factory=lambda: {
        3: {
            'fixed': [0.25, 0.50, 0.75],
            'shifted': [0.20, 0.50, 0.80]
        },
        5: {
            'fixed': [0.10, 0.30, 0.50, 0.70, 0.90],
            'shifted': [0.20, 0.35, 0.50, 0.65, 0.80]
        }
    })
    
    # Noise levels
    sigma_values: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    
    # Spatial discretization
    n_x_values: List[int] = field(default_factory=lambda: [50, 100])
    
    # Time horizon
    T_values: List[float] = field(default_factory=lambda: [0.3, 0.5])
    
    # Temperature parameter for Gibbs posterior
    lambda_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    
    # Loss function scaling parameter
    c_main: float = 1.0
    c_appendix: List[float] = field(default_factory=lambda: [0.5, 2.0])
    
    # Number of segments for piecewise constant º
    m_values: List[int] = field(default_factory=lambda: [3, 5])
    
    # Random seeds (e3 per config)
    seeds: List[int] = field(default_factory=lambda: [101, 202, 303])
    
    # Required knobs (additional parameters)
    n_t_values: List[int] = field(default_factory=lambda: [50, 100])  # Uniform on [0,T]
    delta: float = 0.05  # Certificate confidence
    alpha: float = 1e-3  # Hoeffding bound confidence
    prior_sampling_budget_M: int = 2000  # Increase if underline_Z <= 0
    mc_repeats_R: int = 100  # MC repeats for L_MC estimation
    
    # MCMC parameters
    mcmc_n_steps: int = 10000
    mcmc_n_burn: int = 2000
    mcmc_tau: float = 0.1  # Proposal scale
    mcmc_target_ess: int = 500  # Target ESS per coordinate
    
    # Domain and PDE parameters
    domain: Tuple[float, float] = (0.0, 1.0)
    kappa_min: float = 0.1
    kappa_max: float = 5.0
    
    # Prior parameters (Section D)
    prior_mu_0: float = 0.0
    prior_tau_squared: float = 1.0
    
    # Discretization scheme flags
    use_crank_nicolson: bool = True  # Default CN, False for Backward Euler
    fd_order: int = 2  # Finite difference order
    
    def get_experiment_grid(self, include_appendix: bool = False) -> List[Dict]:
        """
        Generate the full Cartesian product experiment grid.
        
        Returns:
            List of experiment configurations as dictionaries
        """
        experiments = []
        
        # Main experiment grid
        c_values = [self.c_main] if not include_appendix else [self.c_main] + self.c_appendix
        
        # Generate all combinations
        for s, placement_type, sigma, n_x, T, lambda_val, c, m, n_t, seed in itertools.product(
            self.s_values,
            ['fixed', 'shifted'],
            self.sigma_values,
            self.n_x_values,
            self.T_values,
            self.lambda_values,
            c_values,
            self.m_values,
            self.n_t_values,
            self.seeds
        ):
            config = {
                's': s,
                'sensor_positions': self.sensor_placements[s][placement_type],
                'placement_type': placement_type,
                'sigma': sigma,
                'n_x': n_x,
                'T': T,
                'lambda': lambda_val,
                'c': c,
                'm': m,
                'n_t': n_t,
                'seed': seed,
                'delta': self.delta,
                'alpha': self.alpha,
                'M': self.prior_sampling_budget_M,
                'R': self.mc_repeats_R,
                'n': s * n_t,  # Total number of observations
                'Delta_x': 1.0 / n_x,
                'Delta_t': T / (n_t - 1)
            }
            experiments.append(config)
        
        return experiments
    
    def get_baseline_subset(self) -> List[Dict]:
        """
        Get the baseline subset configuration (Section K).
        Run Q_Bayes for n_x=100, both s{3,5} with exact fixed/shifted sets,
        all Ã, one T (e.g., 0.5), both m{3,5}.
        """
        baseline_experiments = []
        
        for s, placement_type, sigma, m, seed in itertools.product(
            self.s_values,
            ['fixed', 'shifted'],
            self.sigma_values,
            self.m_values,
            self.seeds
        ):
            config = {
                's': s,
                'sensor_positions': self.sensor_placements[s][placement_type],
                'placement_type': placement_type,
                'sigma': sigma,
                'n_x': 100,  # Fixed at 100
                'T': 0.5,     # Fixed at 0.5
                'lambda': None,  # Classical Bayes (no tempering)
                'c': self.c_main,
                'm': m,
                'n_t': 100,   # Use finer time grid for baseline
                'seed': seed,
                'delta': self.delta,
                'alpha': self.alpha,
                'M': self.prior_sampling_budget_M,
                'R': self.mc_repeats_R,
                'n': s * 100,
                'Delta_x': 1.0 / 100,
                'Delta_t': 0.5 / 99,
                'is_baseline': True
            }
            baseline_experiments.append(config)
        
        return baseline_experiments
    
    def validate_config(self) -> bool:
        """
        Validate that all configuration parameters are within expected ranges.
        """
        checks = [
            all(s in [3, 5] for s in self.s_values),
            all(0 < sigma < 1 for sigma in self.sigma_values),
            all(n > 0 for n in self.n_x_values),
            all(T > 0 for T in self.T_values),
            all(lambda_val > 0 for lambda_val in self.lambda_values),
            self.kappa_min > 0,
            self.kappa_max > self.kappa_min,
            self.delta > 0 and self.delta < 1,
            self.alpha > 0 and self.alpha < 1,
            self.prior_sampling_budget_M > 0,
            self.mc_repeats_R > 0
        ]
        
        return all(checks)
    
    def log_config(self) -> str:
        """
        Generate a string representation of the configuration for logging.
        """
        config_str = "PAC-Bayes Inverse PDE Experiment Configuration\n"
        config_str += "=" * 50 + "\n"
        config_str += f"Sensors (s): {self.s_values}\n"
        config_str += f"Noise levels (Ã): {self.sigma_values}\n"
        config_str += f"Mesh sizes (n_x): {self.n_x_values}\n"
        config_str += f"Time horizons (T): {self.T_values}\n"
        config_str += f"Temperature params (»): {self.lambda_values}\n"
        config_str += f"Segments (m): {self.m_values}\n"
        config_str += f"Seeds: {self.seeds}\n"
        config_str += f"Certificate confidence (´): {self.delta}\n"
        config_str += f"Hoeffding confidence (±): {self.alpha}\n"
        config_str += f"Prior sampling budget (M): {self.prior_sampling_budget_M}\n"
        config_str += f"MC repeats (R): {self.mc_repeats_R}\n"
        config_str += f"º bounds: [{self.kappa_min}, {self.kappa_max}]\n"
        config_str += "=" * 50 + "\n"
        
        return config_str