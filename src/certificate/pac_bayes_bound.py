"""
Section F: Certificate Computation
PAC-Bayes bound calculation with all components
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.special import logsumexp

class PACBayesCertificate:
    """
    Compute PAC-Bayes generalization certificate.
    
    B_? = L(y,F_h(o)) + (KL(Q_?||P) + ln(1/?))/(?n) + ?_h
    """
    
    def __init__(self, 
                 delta: float = 0.05,
                 alpha: float = 1e-3,
                 M: int = 2000,
                 R: int = 100):
        """
        Initialize certificate calculator.
        
        Args:
            delta: Certificate confidence level
            alpha: Hoeffding bound confidence for Z estimation
            M: Prior sampling budget for Z_? estimation
            R: MC repeats for true risk estimation
        """
        self.delta = delta
        self.alpha = alpha
        self.M = M
        self.R = R
    
    def compute_empirical_term(self, 
                              posterior_samples: np.ndarray,
                              posterior_losses: np.ndarray) -> Dict:
        """
        Section F.1: Compute empirical risk term.
        
        L(y,F_h(o)) evaluated at posterior samples.
        
        Args:
            posterior_samples: MCMC samples from Q_?
            posterior_losses: Loss values at samples
            
        Returns:
            Dictionary with empirical risk statistics
        """
        # Posterior mean of empirical loss
        empirical_mean = np.mean(posterior_losses)
        empirical_std = np.std(posterior_losses)
        empirical_min = np.min(posterior_losses)
        empirical_max = np.max(posterior_losses)
        
        # Check that loss is bounded in (0,1)
        if empirical_min < 0 or empirical_max > 1:
            raise ValueError(f"Loss not in (0,1): [{empirical_min}, {empirical_max}]")
        
        return {
            'empirical_mean': empirical_mean,
            'empirical_std': empirical_std,
            'empirical_min': empirical_min,
            'empirical_max': empirical_max,
            'posterior_losses': posterior_losses
        }
    
    def compute_kl_term(self,
                       posterior_losses: np.ndarray,
                       prior_samples: np.ndarray,
                       prior_losses: np.ndarray,
                       lambda_val: float,
                       n: int) -> Dict:
        """
        Section F.2: Compute KL divergence term (conservative).
        
        KL(Q_?||P) = -ln Z_?(y) - ?n?E_Q[L]
        
        Args:
            posterior_losses: Losses from posterior samples
            prior_samples: Prior samples for Z estimation
            prior_losses: Losses at prior samples
            lambda_val: Temperature parameter
            n: Number of observations
            
        Returns:
            Dictionary with KL estimates
        """
        # Posterior expectation
        E_Q_loss = np.mean(posterior_losses)
        
        # Estimate Z_?(y) = E_{o~P}[exp(-?n?L)]
        log_weights = -lambda_val * n * prior_losses
        
        # Use log-sum-exp for numerical stability
        log_Z_hat = logsumexp(log_weights) - np.log(len(prior_losses))
        Z_hat = np.exp(log_Z_hat)
        
        # Hoeffding lower bound for conservative estimate
        # Z_?(y) e underline_Z = ?_M - sqrt(ln(1/?)/(2M))
        hoeffding_term = np.sqrt(np.log(1/self.alpha) / (2 * self.M))
        underline_Z = Z_hat - hoeffding_term
        
        # Check if underline_Z > 0
        if underline_Z <= 0:
            print(f"Warning: underline_Z = {underline_Z} <= 0")
            print(f"Z_hat = {Z_hat}, hoeffding_term = {hoeffding_term}")
            print("Consider increasing M (prior sampling budget)")
            underline_Z = 1e-10  # Small positive value to continue
        
        log_underline_Z = np.log(underline_Z)
        
        # Conservative KL estimate
        kl_conservative = -log_underline_Z - lambda_val * n * E_Q_loss
        
        # Also compute non-conservative estimate for comparison
        kl_standard = -log_Z_hat - lambda_val * n * E_Q_loss
        
        return {
            'kl_conservative': kl_conservative,
            'kl_standard': kl_standard,
            'Z_hat': Z_hat,
            'underline_Z': underline_Z,
            'log_underline_Z': log_underline_Z,
            'E_Q_loss': E_Q_loss,
            'log_weights': log_weights
        }
    
    def compute_discretization_penalty(self,
                                      kappa_samples: np.ndarray,
                                      solver_h: object,
                                      solver_h2: Optional[object],
                                      loss_fn: object,
                                      y: np.ndarray,
                                      sensor_positions: np.ndarray,
                                      time_grid: np.ndarray) -> Dict:
        """
        Section F.3: Compute discretization penalty ?_h.
        
        Two approaches:
        1. Order bound: ?_h e C??x? + C??t
        2. Refinement test: ?_h = max_o |(y,F_h(o)) - (y,F_{h/2}(o))|
        
        Args:
            kappa_samples: Representative o values for testing
            solver_h: Current mesh solver
            solver_h2: Refined mesh solver (h/2)
            loss_fn: Loss function
            y: Observed data
            sensor_positions: Sensor locations
            time_grid: Time points
            
        Returns:
            Dictionary with discretization penalties
        """
        # Order bound (conservative a priori estimate)
        Delta_x = solver_h.Delta_x
        Delta_t = solver_h.Delta_t
        
        # Standard FD error constants (can be refined based on analysis)
        C1 = 1.0  # Spatial error constant
        C2 = 0.5  # Temporal error constant (0.5 for BE, smaller for CN)
        
        if solver_h.use_crank_nicolson:
            C2 = 0.1  # CN is second-order in time
            eta_order = C1 * Delta_x**2 + C2 * Delta_t**2
        else:
            eta_order = C1 * Delta_x**2 + C2 * Delta_t
        
        # Refinement test (if refined solver provided)
        if solver_h2 is not None:
            eta_refinement = 0.0
            
            for kappa in kappa_samples:
                # Solve on current mesh
                result_h = solver_h.forward_solve(
                    kappa, 
                    sensor_positions=sensor_positions,
                    sensor_times=time_grid
                )
                F_h = result_h['sensor_values']
                loss_h = loss_fn.compute_loss(y, F_h)
                
                # Solve on refined mesh
                result_h2 = solver_h2.forward_solve(
                    kappa,
                    sensor_positions=sensor_positions,
                    sensor_times=time_grid
                )
                F_h2 = result_h2['sensor_values']
                loss_h2 = loss_fn.compute_loss(y, F_h2)
                
                # Update maximum difference
                eta_refinement = max(eta_refinement, abs(loss_h - loss_h2))
            
            # Use the larger (more conservative) estimate
            eta_h = max(eta_order, eta_refinement)
        else:
            eta_h = eta_order
            eta_refinement = None
        
        return {
            'eta_h': eta_h,
            'eta_order': eta_order,
            'eta_refinement': eta_refinement,
            'Delta_x': Delta_x,
            'Delta_t': Delta_t
        }
    
    def compute_final_bound(self,
                          empirical_dict: Dict,
                          kl_dict: Dict,
                          discretization_dict: Dict,
                          lambda_val: float,
                          n: int,
                          lambda_set: Optional[List[float]] = None) -> Dict:
        """
        Section F.4: Compute final PAC-Bayes bound.
        
        B_? = L + (KL + ln(1/?))/(?n) + ?_h
        
        Args:
            empirical_dict: Empirical risk components
            kl_dict: KL divergence components
            discretization_dict: Discretization penalty
            lambda_val: Temperature parameter
            n: Number of observations
            lambda_set: If multiple ? tested, apply union bound
            
        Returns:
            Dictionary with final bound and components
        """
        # Adjust delta for multiple lambda (union bound)
        if lambda_set is not None and len(lambda_set) > 1:
            delta_adjusted = self.delta / len(lambda_set)
            log_delta_term = np.log(len(lambda_set) / self.delta)
        else:
            delta_adjusted = self.delta
            log_delta_term = np.log(1 / self.delta)
        
        # Components
        L_hat = empirical_dict['empirical_mean']
        KL = kl_dict['kl_conservative']
        eta_h = discretization_dict['eta_h']
        
        # PAC-Bayes penalty term
        pac_bayes_penalty = (KL + log_delta_term) / (lambda_val * n)
        
        # Final bound
        B_lambda = L_hat + pac_bayes_penalty + eta_h
        
        return {
            'B_lambda': B_lambda,
            'L_hat': L_hat,
            'KL': KL,
            'pac_bayes_penalty': pac_bayes_penalty,
            'eta_h': eta_h,
            'lambda': lambda_val,
            'n': n,
            'delta': delta_adjusted,
            'components': {
                'empirical_term': L_hat,
                'kl_term': KL / (lambda_val * n),
                'delta_term': log_delta_term / (lambda_val * n),
                'discretization_term': eta_h
            }
        }
    
    def compute_true_risk_mc(self,
                           posterior_samples: np.ndarray,
                           fresh_noise_replicates: np.ndarray,
                           solver: object,
                           loss_fn: object,
                           sensor_positions: np.ndarray,
                           time_grid: np.ndarray) -> Dict:
        """
        Compute true risk L_MC using fresh noise replicates.
        
        L_MC = E_{o~Q_?}[E_{y'~fresh}[(y',F_h(o))]]
        
        Args:
            posterior_samples: MCMC samples from Q_?
            fresh_noise_replicates: Fresh y samples (R, n)
            solver: Forward model solver
            loss_fn: Loss function
            sensor_positions: Sensor locations
            time_grid: Time points
            
        Returns:
            Dictionary with true risk estimates
        """
        n_samples = len(posterior_samples)
        R = len(fresh_noise_replicates)
        
        # Compute loss for each (o, y') pair
        losses_mc = np.zeros((n_samples, R))
        
        for i, kappa in enumerate(posterior_samples):
            # Forward solve once for this o
            result = solver.forward_solve(
                kappa,
                sensor_positions=sensor_positions,
                sensor_times=time_grid
            )
            F_kappa = result['sensor_values']
            
            # Compute loss with each fresh y
            for r in range(R):
                y_fresh = fresh_noise_replicates[r]
                losses_mc[i, r] = loss_fn.compute_loss(y_fresh, F_kappa)
        
        # Average over fresh noise then over posterior
        L_mc = np.mean(losses_mc)
        L_mc_std = np.std(np.mean(losses_mc, axis=1))
        
        return {
            'L_mc': L_mc,
            'L_mc_std': L_mc_std,
            'losses_mc': losses_mc
        }