"""
Phase 2: Block updates and adaptive run length
Extends the adaptive MCMC with sophisticated sampling strategies
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .adaptive_metropolis_hastings import AdaptiveMetropolisHastings

class AdaptiveMetropolisHastingsPhase2(AdaptiveMetropolisHastings):
    """
    Extends Phase 1 with block updates and adaptive run length.
    Designed to achieve target ESS efficiently.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract Phase 2 specific parameters
        self.ess_target = kwargs.pop('ess_target', 400)  # Target ESS per coordinate
        self.chunk_size = kwargs.pop('chunk_size', 5000)  # Sample in chunks
        self.max_steps = kwargs.pop('max_steps', 40000)  # Cap total steps
        self.use_block_updates = kwargs.pop('use_block_updates', True)
        
        super().__init__(*args, **kwargs)
        
        # Block update tracking
        self.block_acceptance_rates = {}
        self.blocks = None
        
    def create_blocks(self) -> List[List[int]]:
        """
        Split parameters into 2-3 blocks for block updates.
        Strategy: group parameters that are likely correlated.
        """
        if self.m <= 3:
            # For m=3, use 2 blocks: [0,1] and [2]
            return [[0, 1], [2]]
        elif self.m == 4:
            # For m=4, use 2 blocks: [0,1] and [2,3]
            return [[0, 1], [2, 3]]
        else:
            # For m=5, use 3 blocks: [0,1], [2,3], [4]
            return [[0, 1], [2, 3], [4]]
    
    def create_block_covariance(self, block: List[int]) -> np.ndarray:
        """
        Extract block-specific covariance matrix.
        """
        if hasattr(self, 'Sigma_proposal') and self.Sigma_proposal is not None:
            # Extract submatrix for this block
            block_cov = self.Sigma_proposal[np.ix_(block, block)]
            
            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(block_cov)
            eigenvals = np.maximum(eigenvals, 1e-6)
            block_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return block_cov
        else:
            # Fallback to diagonal
            return np.eye(len(block)) * (self.tau ** 2)
    
    def block_update_step(self, kappa_current: np.ndarray, 
                         log_p_current: float,
                         blocks: List[List[int]]) -> Tuple[np.ndarray, float, Dict]:
        """
        Update parameters block by block.
        Returns updated state and block-wise acceptance info.
        """
        kappa_new = kappa_current.copy()
        log_p_new = log_p_current
        block_results = {}
        
        for block_idx, block in enumerate(blocks):
            # Propose only for this block
            kappa_proposed = kappa_new.copy()
            
            try:
                if hasattr(self, 'Sigma_proposal') and self.Sigma_proposal is not None:
                    # Use block-specific multivariate proposal
                    block_cov = self.create_block_covariance(block)
                    proposal_delta = self.rng.multivariate_normal(np.zeros(len(block)), block_cov)
                    kappa_proposed[block] = kappa_new[block] + proposal_delta
                else:
                    # Scalar proposals for block
                    kappa_proposed[block] = kappa_new[block] + self.rng.normal(0, self.tau, len(block))
                
            except np.linalg.LinAlgError:
                # Fallback to scalar proposal
                kappa_proposed[block] = kappa_new[block] + self.rng.normal(0, self.tau, len(block))
            
            # Reflect at boundaries for this block
            for i in block:
                if kappa_proposed[i] < self.kappa_min:
                    kappa_proposed[i] = 2 * self.kappa_min - kappa_proposed[i]
                elif kappa_proposed[i] > self.kappa_max:
                    kappa_proposed[i] = 2 * self.kappa_max - kappa_proposed[i]
                kappa_proposed[i] = np.clip(kappa_proposed[i], 
                                           self.kappa_min, self.kappa_max)
            
            # MH acceptance for this block
            log_p_proposed = self.posterior.log_posterior(kappa_proposed)
            log_alpha = log_p_proposed - log_p_new
            
            accepted = False
            if log_alpha > 0 or np.log(self.rng.random()) < log_alpha:
                kappa_new = kappa_proposed
                log_p_new = log_p_proposed
                accepted = True
            
            # Track block-specific acceptance
            block_results[block_idx] = {
                'block': block,
                'accepted': accepted,
                'log_alpha': min(0, log_alpha),  # For diagnostics
                'proposal_norm': np.linalg.norm(kappa_proposed[block] - kappa_current[block])
            }
        
        return kappa_new, log_p_new, block_results
    
    def run_adaptive_length(self, kappa_init: Optional[np.ndarray] = None, 
                          n_burn: int = 2000) -> Dict:
        """
        Run chain with adaptive length based on ESS.
        Collect samples in chunks until ESS >= target per coordinate.
        """
        # Initialize
        if kappa_init is None:
            kappa_init = self.posterior.prior.sample(1, seed=self.rng.randint(10000))[0]
        
        # Burn-in phase (from Phase 1)
        print("Phase 2: Starting burn-in...")
        burn_chain, kappa_current = self.adaptive_burn_in(kappa_init, n_burn)
        
        # Setup for adaptive sampling
        if self.use_block_updates:
            self.blocks = self.create_blocks()
            print(f"Using block updates with {len(self.blocks)} blocks: {self.blocks}")
        else:
            print("Using full parameter updates")
        
        all_samples = []
        chunk_ess_history = []
        block_acceptance_history = []
        log_p_current = self.posterior.log_posterior(kappa_current)
        
        total_steps = 0
        chunk_num = 0
        
        print(f"Starting adaptive sampling (chunks of {self.chunk_size})...")
        print(f"Target: min ESS >= {self.ess_target} per coordinate")
        
        while total_steps < self.max_steps:
            chunk_num += 1
            chunk_samples = np.zeros((self.chunk_size, self.m))
            chunk_block_acceptances = []
            
            # Sample one chunk
            for step in range(self.chunk_size):
                if self.use_block_updates and self.blocks is not None:
                    kappa_current, log_p_current, block_results = self.block_update_step(
                        kappa_current, log_p_current, self.blocks
                    )
                    chunk_block_acceptances.append(block_results)
                else:
                    # Standard full update
                    if hasattr(self, 'Sigma_proposal') and self.Sigma_proposal is not None:
                        kappa_proposed = self.propose_multivariate(kappa_current)
                    else:
                        kappa_proposed = self.propose_scalar(kappa_current)
                    
                    log_p_proposed = self.posterior.log_posterior(kappa_proposed)
                    log_alpha = log_p_proposed - log_p_current
                    
                    if log_alpha > 0 or np.log(self.rng.random()) < log_alpha:
                        kappa_current = kappa_proposed
                        log_p_current = log_p_proposed
                
                chunk_samples[step] = kappa_current
            
            all_samples.append(chunk_samples)
            total_steps += self.chunk_size
            
            # Compute ESS on combined samples
            combined_samples = np.vstack(all_samples)
            ess = self.compute_ess(combined_samples)
            min_ess = np.min(ess)
            mean_ess = np.mean(ess)
            
            # Block acceptance analysis for this chunk
            if chunk_block_acceptances and self.use_block_updates:
                block_acc_rates = {}
                for block_idx in range(len(self.blocks)):
                    acceptances = [res[block_idx]['accepted'] for res in chunk_block_acceptances]
                    block_acc_rates[block_idx] = np.mean(acceptances)
                block_acceptance_history.append(block_acc_rates)
            
            chunk_ess_history.append({
                'chunk': chunk_num,
                'min_ess': min_ess,
                'mean_ess': mean_ess,
                'ess_per_param': ess.tolist(),
                'total_samples': len(combined_samples),
                'cumulative_time': total_steps
            })
            
            print(f"  Chunk {chunk_num}: min_ESS={min_ess:.1f}, "
                  f"mean_ESS={mean_ess:.1f}, total={total_steps}")
            
            # Print block acceptance rates if using blocks
            if chunk_block_acceptances and self.use_block_updates:
                print(f"    Block acceptance rates: {block_acc_rates}")
            
            # Check convergence
            if min_ess >= self.ess_target:
                print(f"? Target ESS reached: {min_ess:.1f} >= {self.ess_target}")
                converged = True
                break
            
            if total_steps >= self.max_steps:
                print(f"? Max steps reached ({self.max_steps}), "
                      f"ESS={min_ess:.1f} < target")
                converged = False
                break
        else:
            converged = False
        
        final_samples = np.vstack(all_samples)
        final_ess = self.compute_ess(final_samples)
        
        # Overall acceptance rate calculation
        if self.use_block_updates and block_acceptance_history:
            overall_block_rates = {}
            for block_idx in range(len(self.blocks)):
                rates = [chunk[block_idx] for chunk in block_acceptance_history]
                overall_block_rates[block_idx] = np.mean(rates)
            overall_acceptance = np.mean(list(overall_block_rates.values()))
        else:
            overall_acceptance = 0.5  # Placeholder for non-block updates
        
        # Efficiency metrics
        efficiency = np.min(final_ess) / total_steps  # ESS per computational step
        
        print(f"\nPhase 2 Summary:")
        print(f"  Total samples: {len(final_samples)}")
        print(f"  Final ESS: min={np.min(final_ess):.1f}, mean={np.mean(final_ess):.1f}")
        print(f"  Efficiency: {efficiency:.4f} ESS/step")
        print(f"  Converged: {converged}")
        
        return {
            'samples': final_samples,
            'burn_chain': burn_chain,
            'n_chunks': chunk_num,
            'total_steps': total_steps,
            'final_ess': final_ess,
            'ess_history': chunk_ess_history,
            'block_acceptance_history': block_acceptance_history,
            'overall_acceptance_rate': overall_acceptance,
            'converged': converged,
            'efficiency': efficiency,
            'blocks_used': self.blocks if self.use_block_updates else None,
            'target_ess': self.ess_target,
            'tau_final': self.tau,
            'Sigma_proposal': self.Sigma_proposal
        }
    
    def run_comparison_study(self, kappa_init: Optional[np.ndarray] = None,
                           n_burn: int = 1000) -> Dict:
        """
        Compare block vs full updates for the same computational budget.
        """
        if kappa_init is None:
            kappa_init = self.posterior.prior.sample(1, seed=42)[0]
        
        results = {}
        
        # Test 1: Block updates
        print("=" * 60)
        print("COMPARISON STUDY: BLOCK UPDATES")
        print("=" * 60)
        
        self.use_block_updates = True
        block_result = self.run_adaptive_length(kappa_init.copy(), n_burn)
        results['block_updates'] = block_result
        
        # Reset sampler state
        self.__init__(self.posterior, self.initial_tau, self.rng.get_state()[1][0])
        
        # Test 2: Full updates  
        print("\n" + "=" * 60)
        print("COMPARISON STUDY: FULL UPDATES")
        print("=" * 60)
        
        self.use_block_updates = False
        full_result = self.run_adaptive_length(kappa_init.copy(), n_burn)
        results['full_updates'] = full_result
        
        # Comparison summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        for method, result in results.items():
            print(f"\n{method.upper()}:")
            print(f"  Min ESS: {np.min(result['final_ess']):.1f}")
            print(f"  Efficiency: {result['efficiency']:.4f}")
            print(f"  Total steps: {result['total_steps']}")
            print(f"  Converged: {result['converged']}")
        
        return results