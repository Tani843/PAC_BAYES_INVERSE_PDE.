"""
Enhanced MCMC with adaptive burn-in tuner
Maintains all PAC-Bayes notation exactly as specified
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any

class AdaptiveMetropolisHastings:
    """
    Adaptive MH sampler that maintains specification notation.
    Addresses the convergence issues identified in the analysis.
    """
    
    def __init__(self, 
                 posterior: Any,  # Accept any posterior object
                 initial_scale: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Initialize with adaptive scaling.
        
        Args:
            posterior: Posterior distribution object  
            initial_scale: Initial ?? (defaults to 0.02*(?_max-?_min))
            seed: Random seed
        """
        self.posterior = posterior
        self.m = posterior.prior.m
        self.kappa_min = posterior.prior.kappa_min
        self.kappa_max = posterior.prior.kappa_max
        
        # Adaptive proposal scale - start more conservative
        if initial_scale is None:
            self.tau = 0.02 * (self.kappa_max - self.kappa_min)
        else:
            self.tau = initial_scale
            
        self.initial_tau = self.tau
        
        # Target acceptance window for burn-in (more conservative)
        self.target_acc_min = 0.25
        self.target_acc_max = 0.35
        
        # RNG
        self.rng = np.random.RandomState(seed)
        
        # Tracking
        self.n_accepted = 0
        self.n_proposed = 0
        self.burn_in_history = []
        self.Sigma_proposal = None
        
    def adaptive_burn_in(self, 
                        kappa_init: np.ndarray,
                        n_burn: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Burn-in with adaptive ? tuning.
        
        Every 50 steps: adjust ? to maintain 25-35% acceptance.
        After 1000 steps: switch to multivariate Gaussian with empirical covariance.
        """
        chain = np.zeros((n_burn, self.m))
        chain[0] = kappa_init
        
        kappa_current = kappa_init.copy()
        log_p_current = self.posterior.log_posterior(kappa_current)
        
        # Phase 1: Adaptive scalar proposal (first 1000 steps)
        adaptation_interval = 50
        
        for step in range(1, min(1000, n_burn)):
            # Standard MH step
            kappa_proposed = self.propose_scalar(kappa_current)
            log_p_proposed = self.posterior.log_posterior(kappa_proposed)
            log_alpha = log_p_proposed - log_p_current
            
            if log_alpha > 0 or np.log(self.rng.random()) < log_alpha:
                kappa_current = kappa_proposed
                log_p_current = log_p_proposed
                self.n_accepted += 1
            
            self.n_proposed += 1
            chain[step] = kappa_current
            
            # Adapt ? every 50 steps
            if step % adaptation_interval == 0 and step > 0:
                acc_rate = self.n_accepted / self.n_proposed
                
                if acc_rate < self.target_acc_min:
                    self.tau *= 0.8  # Decrease step size
                    print(f"  Step {step}: acc={acc_rate:.3f} < {self.target_acc_min}, ??{self.tau:.4f}")
                elif acc_rate > self.target_acc_max:
                    self.tau *= 1.2  # Increase step size
                    print(f"  Step {step}: acc={acc_rate:.3f} > {self.target_acc_max}, ??{self.tau:.4f}")
                
                # Store history
                self.burn_in_history.append({
                    'step': step,
                    'acceptance_rate': acc_rate,
                    'tau': self.tau
                })
                
                # Reset counters for next window
                self.n_accepted = 0
                self.n_proposed = 0
        
        # Phase 2: Multivariate proposal (after step 1000)
        if n_burn > 1000:
            print(f"Switching to multivariate proposal at step 1000")
            
            # Compute empirical covariance from well-mixed samples
            chain_so_far = chain[500:1000]  # Use latter half for better mixing
            if len(chain_so_far) > self.m:
                Sigma_emp = np.cov(chain_so_far.T)
                
                # Regularize covariance matrix
                min_eigenval = 1e-6
                eigenvals, eigenvecs = np.linalg.eigh(Sigma_emp)
                eigenvals = np.maximum(eigenvals, min_eigenval)
                Sigma_emp = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                # Optimal scaling for multivariate normal
                global_scale = (2.38**2) / self.m
                self.Sigma_proposal = global_scale * Sigma_emp
            else:
                # Fallback to diagonal proposal
                scales = np.var(chain_so_far, axis=0)
                scales = np.maximum(scales, (self.tau**2))
                self.Sigma_proposal = np.diag(scales) * (2.38**2) / self.m
            
            # Continue with multivariate proposals
            for step in range(1000, n_burn):
                kappa_proposed = self.propose_multivariate(kappa_current)
                log_p_proposed = self.posterior.log_posterior(kappa_proposed)
                log_alpha = log_p_proposed - log_p_current
                
                if log_alpha > 0 or np.log(self.rng.random()) < log_alpha:
                    kappa_current = kappa_proposed
                    log_p_current = log_p_proposed
                    self.n_accepted += 1
                
                self.n_proposed += 1
                chain[step] = kappa_current
        
        final_acc = self.n_accepted / max(1, self.n_proposed)
        print(f"Burn-in complete: final acceptance={final_acc:.3f}, final ?={self.tau:.4f}")
        
        return chain, kappa_current
    
    def propose_scalar(self, kappa_current: np.ndarray) -> np.ndarray:
        """Scalar proposal with reflection at bounds."""
        kappa_proposed = kappa_current + self.rng.normal(0, self.tau, self.m)
        
        # Reflect at boundaries
        for i in range(self.m):
            if kappa_proposed[i] < self.kappa_min:
                kappa_proposed[i] = 2 * self.kappa_min - kappa_proposed[i]
            elif kappa_proposed[i] > self.kappa_max:
                kappa_proposed[i] = 2 * self.kappa_max - kappa_proposed[i]
            kappa_proposed[i] = np.clip(kappa_proposed[i], self.kappa_min, self.kappa_max)
        
        return kappa_proposed
    
    def propose_multivariate(self, kappa_current: np.ndarray) -> np.ndarray:
        """Multivariate Gaussian proposal with boundary handling."""
        try:
            kappa_proposed = self.rng.multivariate_normal(kappa_current, self.Sigma_proposal)
        except np.linalg.LinAlgError:
            # Fallback to scalar proposal if covariance issues
            return self.propose_scalar(kappa_current)
        
        # Reflect at boundaries
        for i in range(self.m):
            if kappa_proposed[i] < self.kappa_min:
                kappa_proposed[i] = 2 * self.kappa_min - kappa_proposed[i]
            elif kappa_proposed[i] > self.kappa_max:
                kappa_proposed[i] = 2 * self.kappa_max - kappa_proposed[i]
            kappa_proposed[i] = np.clip(kappa_proposed[i], self.kappa_min, self.kappa_max)
        
        return kappa_proposed
    
    def run_chain_adaptive(self, 
                          n_steps: int,
                          n_burn: int,
                          kappa_init: Optional[np.ndarray] = None,
                          thin: int = 1) -> Dict:
        """
        Run full chain with adaptive burn-in.
        """
        # Initialize
        if kappa_init is None:
            kappa_init = self.posterior.prior.sample(1, seed=self.rng.randint(10000))[0]
        
        # Adaptive burn-in
        print("Starting adaptive burn-in...")
        burn_chain, kappa_start = self.adaptive_burn_in(kappa_init, n_burn)
        
        # Main sampling phase
        print(f"Starting main sampling ({n_steps - n_burn} steps)...")
        main_chain = np.zeros((n_steps - n_burn, self.m))
        kappa_current = kappa_start.copy()
        log_p_current = self.posterior.log_posterior(kappa_current)
        
        # Reset counters for main phase
        main_accepted = 0
        main_proposed = 0
        
        for step in range(n_steps - n_burn):
            # Use multivariate proposal if available, otherwise scalar
            if self.Sigma_proposal is not None:
                kappa_proposed = self.propose_multivariate(kappa_current)
            else:
                kappa_proposed = self.propose_scalar(kappa_current)
            
            log_p_proposed = self.posterior.log_posterior(kappa_proposed)
            log_alpha = log_p_proposed - log_p_current
            
            if log_alpha > 0 or np.log(self.rng.random()) < log_alpha:
                kappa_current = kappa_proposed
                log_p_current = log_p_proposed
                main_accepted += 1
            
            main_proposed += 1
            main_chain[step] = kappa_current
        
        # Compute diagnostics
        if thin > 1:
            thinned_chain = main_chain[::thin]
        else:
            thinned_chain = main_chain
            
        ess = self.compute_ess(thinned_chain)
        main_acceptance = main_accepted / max(1, main_proposed)
        
        # Check convergence criteria
        min_ess = np.min(ess)
        converged = min_ess > 200 and main_acceptance > 0.15
        
        print(f"Main sampling complete:")
        print(f"  Acceptance rate: {main_acceptance:.3f}")
        print(f"  ESS: min={min_ess:.1f}, mean={np.mean(ess):.1f}")
        print(f"  Converged: {converged}")
        
        return {
            'samples': thinned_chain,
            'burn_chain': burn_chain,
            'acceptance_rate': main_acceptance,
            'ess': ess,
            'converged': converged,
            'tau_final': self.tau,
            'Sigma_proposal': self.Sigma_proposal,
            'burn_in_history': self.burn_in_history,
            'n_forward_evals': main_proposed + self.n_proposed  # Total evaluations
        }
    
    def compute_ess(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute effective sample size using autocorrelation.
        """
        n_samples, n_params = samples.shape
        ess = np.zeros(n_params)
        
        for i in range(n_params):
            x = samples[:, i]
            
            # Center the series
            x_centered = x - np.mean(x)
            
            # Compute autocorrelation
            n = len(x_centered)
            autocorr = np.correlate(x_centered, x_centered, mode='full')
            autocorr = autocorr[n-1:]  # Take positive lags only
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find integrated autocorrelation time
            # Stop when autocorr becomes negative or very small
            tau_int = 1.0
            for lag in range(1, min(n//4, len(autocorr))):
                if autocorr[lag] <= 0.01:  # Essentially zero
                    break
                tau_int += 2 * autocorr[lag]
            
            # ESS = N / (2 * tau_int)
            ess[i] = n / (2 * tau_int)
            
            # Ensure reasonable bounds
            ess[i] = np.clip(ess[i], 1.0, n)
        
        return ess
    
    def diagnose_posterior(self, kappa_test: Optional[np.ndarray] = None) -> Dict:
        """
        Diagnostic function to understand posterior geometry.
        """
        if kappa_test is None:
            kappa_test = self.posterior.prior.sample(1, seed=42)[0]
        
        # Test log-posterior evaluation
        log_p = self.posterior.log_posterior(kappa_test)
        
        # Test gradient (if available)
        gradient_info = "Not implemented"
        
        # Test proposal scaling
        test_proposals = []
        test_scales = [0.001, 0.01, 0.1, 0.5]
        
        for scale in test_scales:
            old_tau = self.tau
            self.tau = scale
            proposed = self.propose_scalar(kappa_test)
            diff_norm = np.linalg.norm(proposed - kappa_test)
            test_proposals.append({
                'scale': scale,
                'proposal_norm': diff_norm,
                'in_bounds': np.all((proposed >= self.kappa_min) & (proposed <= self.kappa_max))
            })
            self.tau = old_tau
        
        return {
            'test_point': kappa_test,
            'log_posterior': log_p,
            'gradient_info': gradient_info,
            'proposal_tests': test_proposals,
            'domain_bounds': (self.kappa_min, self.kappa_max),
            'dimension': self.m
        }