"""
Section E: MCMC for Q_? and Q_Bayes
Metropolis-Hastings sampler with diagnostics
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from ..inference.gibbs_posterior import GibbsPosterior, ClassicalPosterior

class MetropolisHastings:
    """
    Metropolis-Hastings MCMC sampler for Gibbs and Classical posteriors.
    Implements Section E of the specification.
    """
    
    def __init__(self, 
                 posterior: Union[GibbsPosterior, ClassicalPosterior],
                 proposal_scale: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize MH sampler.
        
        Args:
            posterior: Posterior distribution object
            proposal_scale: Scale ? for proposal distribution
            seed: Random seed for reproducibility
        """
        self.posterior = posterior
        self.proposal_scale = proposal_scale
        self.m = posterior.prior.m  # Dimension
        self.kappa_min = posterior.prior.kappa_min
        self.kappa_max = posterior.prior.kappa_max
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Tracking
        self.n_accepted = 0
        self.n_proposed = 0
    
    def propose(self, kappa_current: np.ndarray) -> np.ndarray:
        """
        Generate proposal respecting box constraints [o_min, o_max].
        
        Uses truncated/reflected Gaussian proposal.
        
        Args:
            kappa_current: Current state
            
        Returns:
            Proposed state
        """
        # Gaussian proposal
        kappa_proposed = kappa_current + self.rng.normal(0, self.proposal_scale, self.m)
        
        # Reflect at boundaries to maintain detailed balance
        for i in range(self.m):
            if kappa_proposed[i] < self.kappa_min:
                kappa_proposed[i] = 2 * self.kappa_min - kappa_proposed[i]
            elif kappa_proposed[i] > self.kappa_max:
                kappa_proposed[i] = 2 * self.kappa_max - kappa_proposed[i]
            
            # Clip if still outside (can happen with large proposals)
            kappa_proposed[i] = np.clip(kappa_proposed[i], self.kappa_min, self.kappa_max)
        
        return kappa_proposed
    
    def metropolis_step(self, kappa_current: np.ndarray, 
                       log_p_current: float) -> Tuple[np.ndarray, float, bool]:
        """
        Single Metropolis-Hastings step.
        
        Args:
            kappa_current: Current state
            log_p_current: Log posterior at current state
            
        Returns:
            Tuple of (new_state, new_log_p, accepted)
        """
        # Propose new state
        kappa_proposed = self.propose(kappa_current)
        
        # Compute log posterior at proposal
        log_p_proposed = self.posterior.log_posterior(kappa_proposed)
        
        # Acceptance ratio (in log space)
        log_alpha = log_p_proposed - log_p_current
        
        # Accept/reject
        if log_alpha > 0 or np.log(self.rng.random()) < log_alpha:
            # Accept
            self.n_accepted += 1
            self.n_proposed += 1
            return kappa_proposed, log_p_proposed, True
        else:
            # Reject
            self.n_proposed += 1
            return kappa_current, log_p_current, False
    
    def run_chain(self, 
                 n_steps: int,
                 n_burn: int,
                 kappa_init: Optional[np.ndarray] = None,
                 thin: int = 1,
                 verbose: bool = True) -> Dict:
        """
        Run MCMC chain with burn-in and thinning.
        
        Args:
            n_steps: Total number of MCMC steps
            n_burn: Number of burn-in steps
            kappa_init: Initial state (sample from prior if None)
            thin: Thinning factor
            verbose: Print progress
            
        Returns:
            Dictionary with:
                - 'samples': Posterior samples
                - 'log_posteriors': Log posterior values
                - 'losses': Loss values at samples
                - 'acceptance_rate': Overall acceptance rate
                - 'ess': Effective sample size per coordinate
                - 'diagnostics': Additional diagnostics
        """
        # Initialize
        if kappa_init is None:
            kappa_init = self.posterior.prior.sample(1, seed=self.rng.randint(10000))[0]
        
        kappa_current = kappa_init.copy()
        log_p_current = self.posterior.log_posterior(kappa_current)
        
        # Storage for samples (after burn-in and thinning)
        n_keep = (n_steps - n_burn) // thin
        samples = np.zeros((n_keep, self.m))
        log_posteriors = np.zeros(n_keep)
        losses = np.zeros(n_keep)
        
        # Full chain for diagnostics
        full_chain = np.zeros((n_steps, self.m))
        acceptance_trace = np.zeros(n_steps, dtype=bool)
        
        # Reset counters
        self.n_accepted = 0
        self.n_proposed = 0
        
        # MCMC loop
        sample_idx = 0
        for step in range(n_steps):
            # Metropolis step
            kappa_current, log_p_current, accepted = self.metropolis_step(
                kappa_current, log_p_current
            )
            
            # Store full chain
            full_chain[step] = kappa_current
            acceptance_trace[step] = accepted
            
            # Store samples after burn-in with thinning
            if step >= n_burn and (step - n_burn) % thin == 0:
                samples[sample_idx] = kappa_current
                log_posteriors[sample_idx] = log_p_current
                
                # Compute loss for this sample
                F_kappa = self.posterior.forward_map(kappa_current)
                loss = self.posterior.loss_fn.compute_empirical_loss(
                    self.posterior.y, F_kappa
                )
                losses[sample_idx] = loss
                
                sample_idx += 1
            
            # Progress
            if verbose and (step + 1) % 1000 == 0:
                acc_rate = self.n_accepted / self.n_proposed
                print(f"Step {step+1}/{n_steps}, Acceptance: {acc_rate:.3f}")
        
        # Compute diagnostics
        acceptance_rate = self.n_accepted / self.n_proposed
        ess_per_coord = self.compute_ess(samples)
        acf = self.compute_acf(samples)
        
        # Check convergence
        converged = self.check_convergence(samples, ess_per_coord, acceptance_rate)
        
        return {
            'samples': samples,
            'log_posteriors': log_posteriors,
            'losses': losses,
            'acceptance_rate': acceptance_rate,
            'ess': ess_per_coord,
            'acf': acf,
            'full_chain': full_chain,
            'acceptance_trace': acceptance_trace,
            'converged': converged,
            'n_forward_evals': self.posterior.n_evals
        }
    
    def compute_ess(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute effective sample size per coordinate.
        
        Args:
            samples: MCMC samples (n_samples, m)
            
        Returns:
            ESS for each coordinate
        """
        n_samples, m = samples.shape
        ess = np.zeros(m)
        
        for j in range(m):
            # Compute autocorrelation
            x = samples[:, j]
            x_centered = x - np.mean(x)
            
            # Autocorrelation function
            acf = np.correlate(x_centered, x_centered, mode='full')
            acf = acf[n_samples-1:] / acf[n_samples-1]
            
            # Find first negative autocorrelation
            first_neg = np.where(acf < 0)[0]
            if len(first_neg) > 0:
                cutoff = first_neg[0]
            else:
                cutoff = len(acf)
            
            # Integrated autocorrelation time
            tau_int = 1 + 2 * np.sum(acf[1:cutoff])
            
            # ESS
            ess[j] = n_samples / tau_int
        
        return ess
    
    def compute_acf(self, samples: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """
        Compute autocorrelation function for diagnostics.
        
        Args:
            samples: MCMC samples
            max_lag: Maximum lag to compute
            
        Returns:
            ACF array (max_lag, m)
        """
        n_samples, m = samples.shape
        max_lag = min(max_lag, n_samples // 4)
        acf = np.zeros((max_lag, m))
        
        for j in range(m):
            x = samples[:, j]
            x_centered = x - np.mean(x)
            
            for lag in range(max_lag):
                if lag == 0:
                    acf[lag, j] = 1.0
                else:
                    acf[lag, j] = np.corrcoef(x_centered[:-lag], x_centered[lag:])[0, 1]
        
        return acf
    
    def check_convergence(self, samples: np.ndarray, 
                         ess: np.ndarray,
                         acceptance_rate: float,
                         min_ess: int = 500) -> Dict:
        """
        Check MCMC convergence criteria.
        
        Args:
            samples: MCMC samples
            ess: Effective sample sizes
            acceptance_rate: Acceptance rate
            min_ess: Minimum required ESS per coordinate
            
        Returns:
            Convergence diagnostics
        """
        diagnostics = {
            'min_ess': np.min(ess),
            'mean_ess': np.mean(ess),
            'acceptance_in_range': 0.2 <= acceptance_rate <= 0.5,
            'ess_sufficient': np.all(ess >= min_ess),
            'converged': False
        }
        
        # Overall convergence
        diagnostics['converged'] = (
            diagnostics['acceptance_in_range'] and 
            diagnostics['ess_sufficient']
        )
        
        return diagnostics