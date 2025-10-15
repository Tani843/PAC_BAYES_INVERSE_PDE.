"""
Section J: Reproducibility Tests
Verify deterministic execution and exact reproducibility
"""

import numpy as np
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import ExperimentConfig
from src.forward_model.heat_equation import HeatEquationSolver
from src.data.data_generator import DataGenerator
from src.inference.loss_functions import BoundedLoss, Prior
from src.inference.gibbs_posterior import GibbsPosterior
from src.mcmc.metropolis_hastings import MetropolisHastings
from src.certificate.pac_bayes_bound import PACBayesCertificate
from src.utils.reproducibility import RandomStateManager, ExperimentTracker


class TestReproducibility:
    """
    Test suite for verifying reproducibility requirements from Section J.
    """
    
    def test_fixed_seeds(self):
        """Test that fixed seeds {101, 202, 303} produce deterministic results."""
        seeds = [101, 202, 303]
        results = []
        
        for seed in seeds:
            # Create config with fixed seed
            config = {
                's': 3,
                'sensor_positions': [0.25, 0.50, 0.75],
                'sigma': 0.1,
                'n_x': 50,
                'n_t': 50,
                'T': 0.5,
                'lambda': 1.0,
                'm': 3,
                'c': 1.0,
                'seed': seed,
                'delta': 0.05,
                'alpha': 1e-3,
                'M': 100,  # Small for testing
                'R': 10,   # Small for testing
                'n': 3 * 50
            }
            
            # Generate data
            data_gen = DataGenerator(config)
            dataset = data_gen.generate_dataset()
            
            # Store result
            results.append(dataset['noisy_data'])
        
        # Verify each seed produces different results
        assert not np.allclose(results[0], results[1])
        assert not np.allclose(results[1], results[2])
        
        # But same seed produces identical results
        config['seed'] = 101
        data_gen2 = DataGenerator(config)
        dataset2 = data_gen2.generate_dataset()
        
        assert np.allclose(results[0], dataset2['noisy_data'])
    
    def test_separate_rng_streams(self):
        """Test that RNG streams are properly separated."""
        rng_manager = RandomStateManager(base_seed=101)
        
        # Get initial states
        data_state1 = rng_manager.get_data_rng().get_state()
        prior_state1 = rng_manager.get_prior_rng().get_state()
        mcmc_state1 = rng_manager.get_mcmc_rng().get_state()
        
        # Use data RNG
        data_samples = rng_manager.get_data_rng().randn(100)
        
        # Check that only data RNG state changed
        data_state2 = rng_manager.get_data_rng().get_state()
        prior_state2 = rng_manager.get_prior_rng().get_state()
        mcmc_state2 = rng_manager.get_mcmc_rng().get_state()
        
        # Data state should have changed
        assert not np.array_equal(data_state1[1], data_state2[1])
        
        # Other states should be unchanged
        assert np.array_equal(prior_state1[1], prior_state2[1])
        assert np.array_equal(mcmc_state1[1], mcmc_state2[1])
        
        # Verify usage log
        assert rng_manager.usage_log['data'] == 2  # Called twice
        assert rng_manager.usage_log['prior'] == 2  # Called twice for state
        assert rng_manager.usage_log['mcmc'] == 2   # Called twice for state
    
    def test_checkpoint_save_load(self):
        """Test that checkpoints can be saved and loaded correctly."""
        from src.utils.reproducibility import CheckpointManager
        
        manager = CheckpointManager('test_checkpoints')
        
        # Create test data
        test_state = {
            'iteration': 42,
            'kappa': np.array([1.0, 2.0, 3.0]),
            'loss': 0.123
        }
        
        test_results = {
            'acceptance_rate': 0.35,
            'ess': [600, 550, 620]
        }
        
        rng_state = {
            'base_seed': 101,
            'data_state': np.random.RandomState(101).get_state(),
            'prior_state': np.random.RandomState(1101).get_state(),
            'mcmc_state': np.random.RandomState(2101).get_state(),
            'usage_log': {'data': 10, 'prior': 5, 'mcmc': 100}
        }
        
        # Save checkpoint
        checkpoint_file = manager.save_checkpoint(
            'test_exp',
            iteration=42,
            state_dict=test_state,
            results=test_results,
            rng_state=rng_state
        )
        
        # Load checkpoint
        loaded = manager.load_checkpoint(checkpoint_file)
        
        # Verify contents
        assert loaded['experiment_id'] == 'test_exp'
        assert loaded['iteration'] == 42
        assert np.allclose(loaded['state_dict']['kappa'], test_state['kappa'])
        assert loaded['results']['acceptance_rate'] == 0.35
        assert loaded['rng_state']['base_seed'] == 101
        
        # Clean up
        checkpoint_file.unlink()
        Path('test_checkpoints').rmdir()
    
    def test_deterministic_mcmc(self):
        """Test that MCMC produces identical chains with same seed."""
        # Setup
        config = {
            's': 3,
            'sensor_positions': [0.25, 0.50, 0.75],
            'sigma': 0.1,
            'n_x': 50,
            'n_t': 50,
            'T': 0.5,
            'lambda': 1.0,
            'm': 3,
            'c': 1.0,
            'seed': 101,
            'n': 150
        }
        
        # Generate fixed data
        np.random.seed(42)
        y = np.random.randn(config['n'])
        
        # Create solver and loss
        solver = HeatEquationSolver(config['n_x'], config['n_t'], config['T'])
        loss_fn = BoundedLoss(config['c'], config['sigma'])
        prior = Prior(config['m'])
        
        # Create posterior
        posterior = GibbsPosterior(
            y=y,
            solver=solver,
            loss_fn=loss_fn,
            prior=prior,
            lambda_val=config['lambda'],
            n=config['n'],
            sensor_positions=np.array(config['sensor_positions']),
            time_grid=np.linspace(0, config['T'], config['n_t'])
        )
        
        # Run MCMC twice with same seed
        sampler1 = MetropolisHastings(posterior, seed=101)
        results1 = sampler1.run_chain(
            n_steps=1000,
            n_burn=200,
            verbose=False
        )
        
        sampler2 = MetropolisHastings(posterior, seed=101)
        results2 = sampler2.run_chain(
            n_steps=1000,
            n_burn=200,
            verbose=False
        )
        
        # Verify identical results
        assert np.allclose(results1['samples'], results2['samples'])
        assert np.allclose(results1['log_posteriors'], results2['log_posteriors'])
        assert results1['acceptance_rate'] == results2['acceptance_rate']
    
    def test_certificate_determinism(self):
        """Test that certificate computation is deterministic."""
        # Create certificate calculator
        cert = PACBayesCertificate(delta=0.05, alpha=1e-3, M=100, R=10)
        
        # Create test data
        np.random.seed(42)
        posterior_samples = np.random.randn(50, 3)
        posterior_losses = np.random.rand(50) * 0.5 + 0.25  # In (0.25, 0.75)
        prior_samples = np.random.randn(100, 3)
        prior_losses = np.random.rand(100) * 0.5 + 0.25
        
        # Compute KL term twice
        kl_dict1 = cert.compute_kl_term(
            posterior_losses,
            prior_samples,
            prior_losses,
            lambda_val=1.0,
            n=150
        )
        
        kl_dict2 = cert.compute_kl_term(
            posterior_losses,
            prior_samples,
            prior_losses,
            lambda_val=1.0,
            n=150
        )
        
        # Verify identical results
        assert kl_dict1['kl_conservative'] == kl_dict2['kl_conservative']
        assert kl_dict1['Z_hat'] == kl_dict2['Z_hat']
        assert kl_dict1['underline_Z'] == kl_dict2['underline_Z']
    
    def test_experiment_tracker(self):
        """Test complete experiment tracking system."""
        config = {
            's': 3,
            'sensor_positions': [0.25, 0.50, 0.75],
            'sigma': 0.1,
            'n_x': 50,
            'seed': 101,
            'lambda': 1.0
        }
        
        # Create tracker
        tracker = ExperimentTracker(config)
        
        # Test RNG streams
        data_rng = tracker.get_rng('data')
        prior_rng = tracker.get_rng('prior')
        mcmc_rng = tracker.get_rng('mcmc')
        
        # Generate some random numbers
        data_vals = data_rng.randn(10)
        prior_vals = prior_rng.randn(10)
        mcmc_vals = mcmc_rng.randn(10)
        
        # Verify different streams produce different values
        assert not np.allclose(data_vals, prior_vals)
        assert not np.allclose(prior_vals, mcmc_vals)
        
        # Test hash computation
        hash1 = tracker.compute_hash(np.array([1, 2, 3]))
        hash2 = tracker.compute_hash(np.array([1, 2, 3]))
        hash3 = tracker.compute_hash(np.array([1, 2, 4]))
        
        assert hash1 == hash2  # Same data -> same hash
        assert hash1 != hash3  # Different data -> different hash
    
    def test_pre_publish_checks(self):
        """Test pre-publish validation checks from Section J."""
        from src.utils.logging import DiagnosticsLogger
        
        logger = DiagnosticsLogger('test_logs')
        
        # Test 1: Loss in (0,1) check
        logger.log_loss_components(
            config={'test': True},
            empirical_loss=0.5,
            bounded_loss=1.5,  # Invalid!
            squared_loss=2.0
        )
        
        assert not logger.loss_diagnostics[-1]['loss_in_bounds']
        
        # Test 2: underline_Z > 0 check
        logger.log_certificate_components(
            config={'s': 3, 'sigma': 0.1, 'lambda': 1.0, 'n': 150, 'delta': 0.05},
            L_hat=0.3,
            KL=2.0,
            Z_hat=0.001,
            underline_Z=-0.0001,  # Invalid!
            eta_h=0.01,
            B_lambda=0.35
        )
        
        assert not logger.certificate_diagnostics[-1]['underline_Z_positive']
        
        # Test 3: Acceptance rate check
        logger.log_chain_diagnostics(
            config={'s': 3, 'sigma': 0.1, 'lambda': 1.0, 'seed': 101},
            chain=np.random.randn(1000, 3),
            acceptance_rate=0.05,  # Too low!
            ess=np.array([600, 550, 620]),
            acf=np.random.randn(50, 3)
        )
        
        assert not logger.chain_diagnostics[-1]['acceptance_in_range']
        
        # Test 4: ESS check
        logger.log_chain_diagnostics(
            config={'s': 3, 'sigma': 0.1, 'lambda': 1.0, 'seed': 101},
            chain=np.random.randn(1000, 3),
            acceptance_rate=0.35,
            ess=np.array([400, 450, 480]),  # Too low!
            acf=np.random.randn(50, 3)
        )
        
        assert not logger.chain_diagnostics[-1]['ess_sufficient']
        
        # Test 5: η_h refinement check
        logger.log_refinement_test(
            n_x_coarse=50,
            n_x_fine=100,
            eta_coarse=0.01,
            eta_fine=0.02  # Should decrease!
        )
        
        assert not logger.convergence_diagnostics[-1]['eta_decreases']
        
        # Clean up
        Path('test_logs').rmdir()


if __name__ == '__main__':
    # Run reproducibility tests
    test = TestReproducibility()
    
    print("Testing fixed seeds...")
    test.test_fixed_seeds()
    print(" Fixed seeds test passed")
    
    print("\nTesting separate RNG streams...")
    test.test_separate_rng_streams()
    print(" RNG streams test passed")
    
    print("\nTesting checkpoint save/load...")
    test.test_checkpoint_save_load()
    print(" Checkpoint test passed")
    
    print("\nTesting MCMC determinism...")
    test.test_deterministic_mcmc()
    print(" MCMC determinism test passed")
    
    print("\nTesting certificate determinism...")
    test.test_certificate_determinism()
    print(" Certificate determinism test passed")
    
    print("\nTesting experiment tracker...")
    test.test_experiment_tracker()
    print(" Experiment tracker test passed")
    
    print("\nTesting pre-publish checks...")
    test.test_pre_publish_checks()
    print(" Pre-publish checks test passed")
    
    print("\n" + "="*50)
    print("All reproducibility tests passed!")
    print("="*50)