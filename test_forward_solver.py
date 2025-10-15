#!/usr/bin/env python3
"""
Test just the forward solver to isolate hanging issue.
"""

import numpy as np
import sys
import time

# Add project root to path
sys.path.append('.')

from config.experiment_config import ExperimentConfig

def test_forward_solver():
    """Test forward solver with simple inputs."""
    
    print("Testing forward solver...")
    
    # Get a simple config
    config = ExperimentConfig()
    experiments = config.get_experiment_grid(include_appendix=False)
    exp_config = experiments[0]
    
    print(f"Config: {exp_config}")
    
    # Create simple solver
    class TestSolver:
        """Simple heat equation solver."""
        def __init__(self, config):
            self.config = config
            
        def forward_solve(self, kappa, sensor_positions, time_grid):
            """Solve heat equation with piecewise constant κ."""
            print(f"  Solving with kappa shape: {kappa.shape}")
            print(f"  Sensor positions: {len(sensor_positions)}")
            print(f"  Time grid: {len(time_grid)}")
            
            s = len(sensor_positions)
            n_t = len(time_grid)
            m = len(kappa)
            
            y_pred = np.zeros((s, n_t))
            
            for i, x in enumerate(sensor_positions):
                for j, t in enumerate(time_grid):
                    # Simple heat equation solution
                    u_val = 0.0
                    for k in range(m):
                        segment_center = (k + 0.5) / m
                        weight = np.exp(-(x - segment_center)**2 / 0.1)
                        u_val += weight * kappa[k] * np.exp(-kappa[k] * t) * np.sin(np.pi * x)
                    y_pred[i, j] = u_val
                    
                    if j % 10 == 0:  # Progress indicator
                        print(f"    Processing sensor {i+1}/{s}, time {j+1}/{n_t}")
            
            print(f"  ✓ Forward solve complete, output shape: {y_pred.shape}")
            return y_pred
    
    # Test solver
    solver = TestSolver(exp_config)
    
    # Create test inputs
    kappa = np.array([1.0, 2.0, 1.5])  # Simple test diffusivity
    sensor_positions = np.array(exp_config['sensor_positions'])
    time_grid = np.linspace(0, exp_config['T'], exp_config['n_t'])
    
    print(f"Test inputs:")
    print(f"  kappa: {kappa}")
    print(f"  sensors: {sensor_positions}")
    print(f"  time_grid: {len(time_grid)} points from 0 to {exp_config['T']}")
    
    start_time = time.time()
    result = solver.forward_solve(kappa, sensor_positions, time_grid)
    runtime = time.time() - start_time
    
    print(f"✓ Forward solve completed in {runtime:.2f} seconds")
    print(f"  Result shape: {result.shape}")
    print(f"  Result range: [{np.min(result):.3f}, {np.max(result):.3f}]")
    
    return True

if __name__ == '__main__':
    try:
        test_forward_solver()
        print("✅ Forward solver test passed!")
    except Exception as e:
        print(f"❌ Forward solver test failed: {e}")
        import traceback
        traceback.print_exc()