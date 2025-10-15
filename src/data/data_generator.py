"""
Section C: Data Generation & Noise
Generate synthetic observations with sensor placement and noise
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..forward_model.heat_equation import HeatEquationSolver

class DataGenerator:
    """
    Generate noisy sensor data for inverse heat equation problem.
    Implements Section C of the specification.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data generator with experiment configuration.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.s = config['s']  # Number of sensors
        self.sensor_positions = np.array(config['sensor_positions'])
        self.n_t = config['n_t']
        self.T = config['T']
        self.sigma = config['sigma']
        self.n_x = config['n_x']
        self.m = config['m']
        self.seed = config['seed']
        
        # Total number of observations
        self.n = self.s * self.n_t
        
        # Time grid (uniform on [0,T])
        self.time_grid = np.linspace(0, self.T, self.n_t)
        
        # Initialize solver
        self.solver = HeatEquationSolver(
            n_x=self.n_x,
            n_t=self.n_t,
            T=self.T,
            kappa_min=config.get('kappa_min', 0.1),
            kappa_max=config.get('kappa_max', 5.0),
            use_crank_nicolson=config.get('use_crank_nicolson', True)
        )
        
        # Random number generator with fixed seed
        self.rng = np.random.RandomState(self.seed)
    
    def generate_true_kappa(self, kappa_star: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate or use provided true conductivity field o*.
        
        Args:
            kappa_star: If provided, use this; otherwise generate random
            
        Returns:
            True conductivity values for m segments
        """
        if kappa_star is not None:
            return kappa_star
        
        # Generate random piecewise constant field
        # Sample from uniform distribution in [kappa_min, kappa_max]
        kappa_min = self.config.get('kappa_min', 0.1)
        kappa_max = self.config.get('kappa_max', 5.0)
        
        kappa_true = self.rng.uniform(kappa_min, kappa_max, self.m)
        
        return kappa_true
    
    def generate_noiseless_data(self, kappa_star: np.ndarray) -> np.ndarray:
        """
        Generate noiseless observations F_h(o*).
        
        Args:
            kappa_star: True conductivity field
            
        Returns:
            Noiseless sensor observations
        """
        # Solve forward problem
        result = self.solver.forward_solve(
            kappa=kappa_star,
            sensor_positions=self.sensor_positions,
            sensor_times=self.time_grid
        )
        
        return result['sensor_values']
    
    def add_noise(self, noiseless_data: np.ndarray, 
                 fresh_seed: Optional[int] = None) -> np.ndarray:
        """
        Add Gaussian noise to observations.
        
        y_ij = u(x_i, t_j; o*) + ?_ij
        ?_ij ~ iid N(0, ??)
        
        Args:
            noiseless_data: Clean sensor observations
            fresh_seed: Optional seed for reproducible fresh noise
            
        Returns:
            Noisy observations
        """
        if fresh_seed is not None:
            noise_rng = np.random.RandomState(fresh_seed)
        else:
            noise_rng = self.rng
        
        # Generate iid Gaussian noise
        noise = noise_rng.normal(0, self.sigma, size=noiseless_data.shape)
        
        return noiseless_data + noise
    
    def generate_dataset(self, kappa_star: Optional[np.ndarray] = None) -> Dict:
        """
        Generate complete dataset with true parameters and noisy observations.
        
        Args:
            kappa_star: True conductivity (generated if not provided)
            
        Returns:
            Dictionary containing:
                - 'kappa_star': True conductivity
                - 'noiseless_data': F_h(o*)
                - 'noisy_data': y = F_h(o*) + ?
                - 'sensor_positions': Sensor x-locations
                - 'time_grid': Time points
                - 'config': Full configuration
        """
        # Generate or use true conductivity
        kappa_true = self.generate_true_kappa(kappa_star)
        
        # Generate noiseless data
        noiseless = self.generate_noiseless_data(kappa_true)
        
        # Add noise
        noisy = self.add_noise(noiseless)
        
        return {
            'kappa_star': kappa_true,
            'noiseless_data': noiseless,
            'noisy_data': noisy,
            'sensor_positions': self.sensor_positions,
            'time_grid': self.time_grid,
            'n': self.n,
            's': self.s,
            'n_t': self.n_t,
            'sigma': self.sigma,
            'config': self.config
        }
    
    def generate_fresh_noise_replicates(self, noiseless_data: np.ndarray, 
                                       R: int = 100) -> np.ndarray:
        """
        Generate R fresh noise replicates for MC risk estimation.
        Used for computing L_MC in experiments.
        
        Args:
            noiseless_data: Clean observations F_h(o*)
            R: Number of replicates
            
        Returns:
            Array of shape (R, n) with fresh noisy observations
        """
        replicates = np.zeros((R, len(noiseless_data)))
        
        # Use different seeds for each replicate
        for r in range(R):
            fresh_seed = self.seed + 1000 + r  # Ensure different from main seed
            replicates[r] = self.add_noise(noiseless_data, fresh_seed=fresh_seed)
        
        return replicates
    
    def get_sensor_design_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the sensor design in matrix form for analysis.
        
        Returns:
            Tuple of (X, T) where:
                - X: sensor positions (s,)
                - T: time points (n_t,)
        """
        return self.sensor_positions, self.time_grid
    
    def log_data_summary(self, dataset: Dict) -> str:
        """
        Generate summary statistics for logging.
        
        Args:
            dataset: Generated dataset dictionary
            
        Returns:
            Summary string
        """
        summary = "Data Generation Summary\n"
        summary += "=" * 40 + "\n"
        summary += f"Sensors (s): {self.s}\n"
        summary += f"Sensor positions: {self.sensor_positions}\n"
        summary += f"Time points (n_t): {self.n_t}\n"
        summary += f"Total observations (n): {self.n}\n"
        summary += f"Noise level (?): {self.sigma}\n"
        summary += f"True o*: {dataset['kappa_star']}\n"
        summary += f"Data range (noiseless): [{dataset['noiseless_data'].min():.4f}, "
        summary += f"{dataset['noiseless_data'].max():.4f}]\n"
        summary += f"Data range (noisy): [{dataset['noisy_data'].min():.4f}, "
        summary += f"{dataset['noisy_data'].max():.4f}]\n"
        summary += f"SNR: {np.std(dataset['noiseless_data']) / self.sigma:.2f}\n"
        summary += "=" * 40 + "\n"
        
        return summary