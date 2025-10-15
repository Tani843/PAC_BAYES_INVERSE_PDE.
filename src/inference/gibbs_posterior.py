"""
Section D: Gibbs/Generalized Posterior
Q_?(do)  exp(-?n?L(y,F(o)))?P(do)
"""

import numpy as np
from typing import Dict, Optional, Callable
from .loss_functions import BoundedLoss, Prior
from ..forward_model.heat_equation import HeatEquationSolver

class GibbsPosterior:
    """
    Gibbs (tempered/generalized) posterior for PAC-Bayes certification.
    
    Q_?(do)  exp(-?n?L(y,F(o)))?P(do)
    """
    
    def __init__(self, 
                 y: np.ndarray,
                 solver: HeatEquationSolver,
                 loss_fn: BoundedLoss,
                 prior: Prior,
                 lambda_val: float,
                 n: int,
                 sensor_positions: np.ndarray,
                 time_grid: np.ndarray):
        """
        Initialize Gibbs posterior.
        
        Args:
            y: Observed data
            solver: Forward model solver
            loss_fn: Bounded loss function
            prior: Prior distribution
            lambda_val: Temperature parameter
            n: Number of observations
            sensor_positions: Sensor x-locations
            time_grid: Time points
        """
        self.y = y
        self.solver = solver
        self.loss_fn = loss_fn
        self.prior = prior
        self.lambda_val = lambda_val
        self.n = n
        self.sensor_positions = sensor_positions
        self.time_grid = time_grid
        
        # Cache for forward evaluations
        self.eval_cache = {}
        self.n_evals = 0
    
    def forward_map(self, kappa: np.ndarray) -> np.ndarray:
        """
        Evaluate forward model F_h(o) at sensors.
        
        Args:
            kappa: Conductivity parameters
            
        Returns:
            Model predictions at sensors
        """
        # Check cache
        kappa_key = tuple(kappa)
        if kappa_key in self.eval_cache:
            return self.eval_cache[kappa_key]
        
        # Solve forward problem
        result = self.solver.forward_solve(
            kappa=kappa,
            sensor_positions=self.sensor_positions,
            sensor_times=self.time_grid
        )
        
        F_kappa = result['sensor_values']
        
        # Cache result
        self.eval_cache[kappa_key] = F_kappa
        self.n_evals += 1
        
        return F_kappa
    
    def log_likelihood(self, kappa: np.ndarray) -> float:
        """
        Compute tempered log-likelihood: -?n?L(y,F(o)).
        
        Args:
            kappa: Conductivity parameters
            
        Returns:
            Log-likelihood value
        """
        # Forward evaluation
        F_kappa = self.forward_map(kappa)
        
        # Empirical loss
        loss = self.loss_fn.compute_empirical_loss(self.y, F_kappa)
        
        # Tempered log-likelihood
        return -self.lambda_val * self.n * loss
    
    def log_posterior(self, kappa: np.ndarray) -> float:
        """
        Compute unnormalized log posterior.
        
        log Q_?(o) = -?n?L(y,F(o)) + log P(o) + const
        
        Args:
            kappa: Conductivity parameters
            
        Returns:
            Unnormalized log posterior
        """
        log_prior = self.prior.log_pdf(kappa)
        
        # Return -inf if outside bounds
        if log_prior == -np.inf:
            return -np.inf
        
        log_lik = self.log_likelihood(kappa)
        
        return log_lik + log_prior
    
    def compute_partition_function_estimate(self, M: int = 2000, 
                                           seed: Optional[int] = None) -> Dict:
        """
        Estimate partition function Z_?(y) using prior samples.
        
        Z_?(y) = E_{o~P}[exp(-?n?L(y,F(o)))]
        
        Args:
            M: Number of prior samples
            seed: Random seed
            
        Returns:
            Dictionary with Z estimates and samples
        """
        # Sample from prior
        prior_samples = self.prior.sample(M, seed=seed)
        
        # Compute log weights
        log_weights = np.zeros(M)
        for i in range(M):
            kappa = prior_samples[i]
            F_kappa = self.forward_map(kappa)
            loss = self.loss_fn.compute_empirical_loss(self.y, F_kappa)
            log_weights[i] = -self.lambda_val * self.n * loss
        
        # Use log-sum-exp for numerical stability
        max_log_w = np.max(log_weights)
        Z_est = np.exp(max_log_w) * np.mean(np.exp(log_weights - max_log_w))
        
        # Log-space estimate
        log_Z_est = max_log_w + np.log(np.mean(np.exp(log_weights - max_log_w)))
        
        return {
            'Z_est': Z_est,
            'log_Z_est': log_Z_est,
            'log_weights': log_weights,
            'prior_samples': prior_samples
        }
    
    def compute_kl_divergence(self, posterior_samples: np.ndarray,
                             posterior_losses: np.ndarray,
                             Z_dict: Dict) -> float:
        """
        Compute KL divergence KL(Q_?||P) using Gibbs identity.
        
        KL(Q_?||P) = -ln Z_?(y) - ?n?E_Q[L]
        
        Args:
            posterior_samples: MCMC samples from Q_?
            posterior_losses: Loss values for samples
            Z_dict: Partition function estimates
            
        Returns:
            KL divergence estimate
        """
        # Posterior expectation of loss
        E_Q_loss = np.mean(posterior_losses)
        
        # KL using Gibbs identity
        kl = -Z_dict['log_Z_est'] - self.lambda_val * self.n * E_Q_loss
        
        return kl


class ClassicalPosterior(GibbsPosterior):
    """
    Classical Bayesian posterior (baseline).
    
    Q_Bayes(do)  exp(-||y-F(o)||?/(2??))?P(do)
    """
    
    def __init__(self, 
                 y: np.ndarray,
                 solver: HeatEquationSolver,
                 prior: Prior,
                 sigma: float,
                 sensor_positions: np.ndarray,
                 time_grid: np.ndarray):
        """
        Initialize classical Bayesian posterior.
        
        Args:
            y: Observed data
            solver: Forward model solver
            prior: Prior distribution
            sigma: Noise standard deviation
            sensor_positions: Sensor locations
            time_grid: Time points
        """
        # Initialize with dummy loss for parent class
        dummy_loss = BoundedLoss(c=1.0, sigma=sigma)
        super().__init__(y, solver, dummy_loss, prior, 
                        lambda_val=1.0, n=len(y),
                        sensor_positions=sensor_positions,
                        time_grid=time_grid)
        
        self.sigma = sigma
        self.sigma2 = sigma ** 2
    
    def log_likelihood(self, kappa: np.ndarray) -> float:
        """
        Compute classical Gaussian log-likelihood.
        
        Args:
            kappa: Conductivity parameters
            
        Returns:
            Log-likelihood value
        """
        # Forward evaluation
        F_kappa = self.forward_map(kappa)
        
        # Squared error
        squared_error = np.sum((self.y - F_kappa) ** 2)
        
        # Gaussian log-likelihood
        log_lik = -squared_error / (2 * self.sigma2)
        
        return log_lik