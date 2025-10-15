"""
Classical Bayesian posterior implementation
"""
import numpy as np

class ClassicalPosterior:
    """
    Classical Bayesian posterior with Gaussian likelihood.
    Uses EXACT same forward operator as GibbsPosterior.
    """
    
    def __init__(self, y, solver, sigma, prior, sensor_positions, time_grid, n):
        self.y = y
        self.solver = solver
        self.sigma = sigma
        self.prior = prior
        self.sensor_positions = sensor_positions
        self.time_grid = time_grid
        self.n = n
        self.m = prior.m
        
    def log_posterior(self, kappa):
        """
        Classical log posterior: -||y-F(kappa)||^2/(2*sigma^2) + log p(kappa)
        """
        # Use EXACT same forward_map as GibbsPosterior
        F_kappa = self.forward_map(kappa)
        
        # Gaussian log-likelihood
        residual = self.y - F_kappa
        log_likelihood = -0.5 * np.sum(residual**2) / (self.sigma**2)
        
        # Use EXACT same prior.log_pdf as GibbsPosterior
        log_prior = self.prior.log_pdf(kappa)
        
        return log_likelihood + log_prior
    
    def forward_map(self, kappa):
        """
        IDENTICAL to GibbsPosterior.forward_map
        Uses solver.forward_solve with sensor restrictions
        """
        F_kappa = self.solver.forward_solve(
            kappa, 
            self.sensor_positions, 
            self.time_grid
        )
        # Ensure flattened to match y shape
        return F_kappa.flatten() if F_kappa.ndim > 1 else F_kappa