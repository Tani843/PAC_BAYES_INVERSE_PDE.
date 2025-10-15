"""
Section J: Reproducibility
Seed management, deterministic execution, and experiment tracking
"""

import numpy as np
import random
import hashlib
import json
import pickle
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import colorlog
import yaml

class RandomStateManager:
    """
    Manages separate RNG streams for reproducibility.
    
    Three independent streams as specified:
    1. Data noise generation
    2. Prior sampling for Z_M
    3. MCMC proposals
    """
    
    def __init__(self, base_seed: int):
        """
        Initialize RNG manager with base seed.
        
        Args:
            base_seed: Base seed (101, 202, or 303)
        """
        self.base_seed = base_seed
        
        # Create separate RNG streams with deterministic offsets
        self.data_rng = np.random.RandomState(base_seed)
        self.prior_rng = np.random.RandomState(base_seed + 1000)
        self.mcmc_rng = np.random.RandomState(base_seed + 2000)
        
        # Python's random module for complete coverage
        random.seed(base_seed + 3000)
        
        # Track usage for debugging
        self.usage_log = {
            'data': 0,
            'prior': 0,
            'mcmc': 0
        }
    
    def get_data_rng(self) -> np.random.RandomState:
        """Get RNG for data noise generation."""
        self.usage_log['data'] += 1
        return self.data_rng
    
    def get_prior_rng(self) -> np.random.RandomState:
        """Get RNG for prior sampling."""
        self.usage_log['prior'] += 1
        return self.prior_rng
    
    def get_mcmc_rng(self) -> np.random.RandomState:
        """Get RNG for MCMC proposals."""
        self.usage_log['mcmc'] += 1
        return self.mcmc_rng
    
    def get_state_dict(self) -> Dict:
        """Get current state of all RNGs for checkpointing."""
        return {
            'base_seed': self.base_seed,
            'data_state': self.data_rng.get_state(),
            'prior_state': self.prior_rng.get_state(),
            'mcmc_state': self.mcmc_rng.get_state(),
            'usage_log': self.usage_log.copy()
        }
    
    def set_state_dict(self, state_dict: Dict):
        """Restore RNG states from checkpoint."""
        self.base_seed = state_dict['base_seed']
        self.data_rng.set_state(state_dict['data_state'])
        self.prior_rng.set_state(state_dict['prior_state'])
        self.mcmc_rng.set_state(state_dict['mcmc_state'])
        self.usage_log = state_dict['usage_log'].copy()


class ExperimentLogger:
    """
    Comprehensive logging for experiment tracking and reproducibility.
    """
    
    def __init__(self, log_dir: str = 'results/logs', experiment_id: Optional[str] = None):
        """
        Initialize logger with colored console output and file logging.
        
        Args:
            log_dir: Directory for log files
            experiment_id: Unique experiment identifier
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_id = f"exp_{timestamp}"
        else:
            self.experiment_id = experiment_id
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Experiment metadata
        self.metadata = {
            'experiment_id': self.experiment_id,
            'start_time': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'package_versions': self._get_package_versions()
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup colored console and file logging."""
        # Create logger
        logger = logging.getLogger(self.experiment_id)
        logger.setLevel(logging.DEBUG)
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white'
            }
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash for tracking."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return None
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        versions = {}
        packages = ['numpy', 'scipy', 'matplotlib', 'pandas']
        
        for pkg in packages:
            try:
                module = __import__(pkg)
                versions[pkg] = module.__version__
            except:
                versions[pkg] = 'not installed'
        
        return versions
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
        self.logger.info("Experiment Configuration:")
        self.logger.info("=" * 60)
        
        for key, value in config.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 5:
                    self.logger.info(f"{key}: {type(value).__name__} of length {len(value)}")
                else:
                    self.logger.info(f"{key}: {value}")
            else:
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 60)
        
        # Save config to file
        config_file = self.log_dir / f"{self.experiment_id}_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def log_iteration(self, iteration: int, metrics: Dict):
        """Log iteration metrics."""
        msg = f"Iteration {iteration}: "
        msg += ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                         for k, v in metrics.items()])
        self.logger.info(msg)
    
    def log_convergence(self, converged: bool, diagnostics: Dict):
        """Log convergence status."""
        if converged:
            self.logger.info(" Convergence achieved!")
        else:
            self.logger.warning(" Convergence not achieved")
        
        for key, value in diagnostics.items():
            self.logger.debug(f"  {key}: {value}")
    
    def save_metadata(self):
        """Save experiment metadata."""
        self.metadata['end_time'] = datetime.now().isoformat()
        
        metadata_file = self.log_dir / f"{self.experiment_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)


class CheckpointManager:
    """
    Manages checkpointing for experiment resumption.
    """
    
    def __init__(self, checkpoint_dir: str = 'results/checkpoints'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       experiment_id: str,
                       iteration: int,
                       state_dict: Dict,
                       results: Dict,
                       rng_state: Optional[Dict] = None):
        """
        Save checkpoint with all necessary state.
        
        Args:
            experiment_id: Unique experiment ID
            iteration: Current iteration number
            state_dict: Model/algorithm state
            results: Current results
            rng_state: RNG states for exact resumption
        """
        checkpoint = {
            'experiment_id': experiment_id,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'state_dict': state_dict,
            'results': results,
            'rng_state': rng_state
        }
        
        # Create checkpoint filename with iteration
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}_iter_{iteration}.pkl"
        
        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save a "latest" symlink for easy access
        latest_file = self.checkpoint_dir / f"{experiment_id}_latest.pkl"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(checkpoint_file.name)
        
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return checkpoint
    
    def find_latest_checkpoint(self, experiment_id: str) -> Optional[Path]:
        """
        Find the latest checkpoint for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Path to latest checkpoint or None
        """
        latest_file = self.checkpoint_dir / f"{experiment_id}_latest.pkl"
        
        if latest_file.exists():
            return latest_file
        
        # Fall back to finding highest iteration
        pattern = f"{experiment_id}_iter_*.pkl"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if checkpoints:
            # Sort by iteration number
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            return checkpoints[-1]
        
        return None
    
    def cleanup_old_checkpoints(self, experiment_id: str, keep_last: int = 3):
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            experiment_id: Experiment ID
            keep_last: Number of recent checkpoints to keep
        """
        pattern = f"{experiment_id}_iter_*.pkl"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) > keep_last:
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()


class ExperimentTracker:
    """
    Complete experiment tracking system combining all reproducibility components.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize experiment tracker.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.seed = config['seed']
        
        # Initialize components
        self.rng_manager = RandomStateManager(self.seed)
        self.logger = ExperimentLogger()
        self.checkpoint_manager = CheckpointManager()
        
        # Log initial setup
        self.logger.log_config(config)
        self.logger.logger.info(f"Initialized with seed: {self.seed}")
        
        # Track experiment progress
        self.metrics_history = []
        self.current_iteration = 0
    
    def get_rng(self, stream: str) -> np.random.RandomState:
        """
        Get appropriate RNG stream.
        
        Args:
            stream: 'data', 'prior', or 'mcmc'
            
        Returns:
            RNG for specified stream
        """
        if stream == 'data':
            return self.rng_manager.get_data_rng()
        elif stream == 'prior':
            return self.rng_manager.get_prior_rng()
        elif stream == 'mcmc':
            return self.rng_manager.get_mcmc_rng()
        else:
            raise ValueError(f"Unknown RNG stream: {stream}")
    
    def save_state(self, results: Dict, state_dict: Optional[Dict] = None):
        """
        Save current experiment state.
        
        Args:
            results: Current results
            state_dict: Additional state to save
        """
        # Get RNG states
        rng_state = self.rng_manager.get_state_dict()
        
        # Save checkpoint
        checkpoint_file = self.checkpoint_manager.save_checkpoint(
            experiment_id=self.logger.experiment_id,
            iteration=self.current_iteration,
            state_dict=state_dict or {},
            results=results,
            rng_state=rng_state
        )
        
        self.logger.logger.info(f"Saved checkpoint: {checkpoint_file}")
        
        # Clean up old checkpoints
        self.checkpoint_manager.cleanup_old_checkpoints(
            self.logger.experiment_id,
            keep_last=3
        )
    
    def load_state(self, checkpoint_path: str):
        """
        Load experiment state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Restore RNG states
        if checkpoint['rng_state']:
            self.rng_manager.set_state_dict(checkpoint['rng_state'])
        
        self.current_iteration = checkpoint['iteration']
        
        self.logger.logger.info(f"Loaded checkpoint from iteration {self.current_iteration}")
        
        return checkpoint
    
    def compute_hash(self, data: Any) -> str:
        """
        Compute deterministic hash for data verification.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex hash string
        """
        # Convert to bytes
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, (list, dict)):
            data_bytes = json.dumps(data, sort_keys=True).encode()
        else:
            data_bytes = str(data).encode()
        
        # Compute SHA256 hash
        return hashlib.sha256(data_bytes).hexdigest()
    
    def verify_reproducibility(self, results1: Dict, results2: Dict) -> bool:
        """
        Verify that two result sets are identical.
        
        Args:
            results1: First result set
            results2: Second result set
            
        Returns:
            True if results match
        """
        hash1 = self.compute_hash(results1)
        hash2 = self.compute_hash(results2)
        
        match = (hash1 == hash2)
        
        if match:
            self.logger.logger.info(" Reproducibility verified: Results match exactly")
        else:
            self.logger.logger.warning(" Reproducibility check failed: Results differ")
            self.logger.logger.debug(f"Hash 1: {hash1}")
            self.logger.logger.debug(f"Hash 2: {hash2}")
        
        return match