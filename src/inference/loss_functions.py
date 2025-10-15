"""
Section D: Loss Functions and Prior
Bounded loss functions for PAC-Bayes certification
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm, truncnorm

class BoundedLoss:
    """
    Sigmoid-bounded squared loss for PAC-Bayes compatibility.
    
    (y,F(o)) = (1/n) ?_{i,j} ?((y_ij - F(o)_ij)? / (c???))
    where ?(z) = 1/(1 + e^{-z})
    """
    
    def __init__(self, c: float = 1.0, sigma: float = 0.1):
        """
        Initialize bounded loss function.
        
        Args:
            c: Scaling parameter for residuals
            sigma: Noise standard deviation
        """
        self.c = c
        self.sigma = sigma
        self.sigma2 = sigma ** 2
    
    def phi(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid function ?(z) = 1/(1 + e^{-z}).
        Ensures output in (0,1) for PAC-Bayes.
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid-transformed values
        """
        # Numerically stable sigmoid
        return np.where(z >= 0,
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def compute_loss(self, y: np.ndarray, F_kappa: np.ndarray) -> float:
        """
        Compute bounded loss (y,F(o)).
        
        Args:
            y: Observed data
            F_kappa: Model predictions F(o)
            
        Returns:
            Bounded loss value in (0,1)
        """
        n = len(y)
        
        # Squared residuals scaled by c???
        squared_residuals = (y - F_kappa) ** 2
        scaled_residuals = squared_residuals / (self.c * self.sigma2)
        
        # Apply sigmoid and average
        loss = np.mean(self.phi(scaled_residuals))
        
        return loss
    
    def compute_empirical_loss(self, y: np.ndarray, F_kappa: np.ndarray) -> float:
        """
        Compute empirical loss L(y,F(o)) - same as compute_loss but explicit naming.
        
        Args:
            y: Observed data
            F_kappa: Model predictions
            
        Returns:
            Empirical loss value
        """
        return self.compute_loss(y, F_kappa)
    
    def compute_true_loss(self, F_kappa: np.ndarray, 
                         fresh_noise_replicates: np.ndarray) -> float:
        """
        Compute true loss L(y,F(o)) by averaging over fresh noise.
        
        Args:
            F_kappa: Model predictions
            fresh_noise_replicates: Array of shape (R, n) with R fresh y samples
            
        Returns:
            True loss estimate
        """
        R = fresh_noise_replicates.shape[0]
        losses = np.zeros(R)
        
        for r in range(R):
            y_fresh = fresh_noise_replicates[r]
            losses[r] = self.compute_loss(y_fresh, F_kappa)
        
        return np.mean(losses)
    
    def compute_squared_loss(self, y: np.ndarray, F_kappa: np.ndarray) -> float:
        """
        Compute standard squared loss for comparison (unbounded).
        
        Args:
            y: Observed data
            F_kappa: Model predictions
            
        Returns:
            Mean squared error
        """
        return np.mean((y - F_kappa) ** 2)


class Prior:
    """
    Prior distribution for conductivity parameters.
    P(o) = ?_{r=1}^m N(o_r | ?_0, ??) truncated to [o_min, o_max]
    """
    
    def __init__(self, m: int, mu_0: float = 0.0, tau2: float = 1.0,
                 kappa_min: float = 0.1, kappa_max: float = 5.0):
        """
        Initialize prior distribution.
        
        Args:
            m: Number of segments
            mu_0: Prior mean
            tau2: Prior variance
            kappa_min: Lower bound for truncation
            kappa_max: Upper bound for truncation
        """
        self.m = m
        self.mu_0 = mu_0
        self.tau2 = tau2
        self.tau = np.sqrt(tau2)
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        
        # Compute truncation parameters for scipy
        self.a = (kappa_min - mu_0) / self.tau
        self.b = (kappa_max - mu_0) / self.tau
    
    def sample(self, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample from the prior distribution.
        
        Args:
            n_samples: Number of samples to draw
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, m) with prior samples
        """
        if seed is not None:
            np.random.seed(seed)
        
        samples = np.zeros((n_samples, self.m))
        
        for i in range(n_samples):
            for r in range(self.m):
                # Sample from truncated normal
                samples[i, r] = truncnorm.rvs(
                    self.a, self.b, 
                    loc=self.mu_0, 
                    scale=self.tau
                )
        
        return samples
    
    def log_pdf(self, kappa: np.ndarray) -> float:
        """
        Compute log prior density log P(o).
        
        Args:
            kappa: Parameter vector (m,)
            
        Returns:
            Log prior density
        """
        # Check bounds
        if np.any(kappa < self.kappa_min) or np.any(kappa > self.kappa_max):
            return -np.inf
        
        log_p = 0.0
        for r in range(self.m):
            # Log PDF of truncated normal
            log_p += truncnorm.logpdf(
                kappa[r], 
                self.a, self.b,
                loc=self.mu_0, 
                scale=self.tau
            )
        
        return log_p
    
    def pdf(self, kappa: np.ndarray) -> float:
        """
        Compute prior density P(o).
        
        Args:
            kappa: Parameter vector
            
        Returns:
            Prior density
        """
        return np.exp(self.log_pdf(kappa))