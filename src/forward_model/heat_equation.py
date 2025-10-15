"""
Section B: Forward Model F_h
Heat equation solver with finite difference discretization
Domain: ?=[0,1], Dirichlet BC: u(0,t)=u(1,t)=0, horizon T
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
import warnings

class HeatEquationSolver:
    """
    Finite difference solver for 1D heat equation with spatially-varying conductivity.
    
    u/t = ?(?(x)u) + f(x,t)
    u(0,t) = u(1,t) = 0
    u(x,0) = u_0(x)
    """
    
    def __init__(self, 
                 n_x: int, 
                 n_t: int,
                 T: float,
                 kappa_min: float = 0.1,
                 kappa_max: float = 5.0,
                 use_crank_nicolson: bool = True,
                 fd_order: int = 2):
        """
        Initialize heat equation solver.
        
        Args:
            n_x: Number of spatial grid points (including boundaries)
            n_t: Number of time points
            T: Time horizon
            kappa_min: Minimum conductivity value
            kappa_max: Maximum conductivity value
            use_crank_nicolson: If True, use Crank-Nicolson; else Backward Euler
            fd_order: Finite difference order (2 for second-order)
        """
        self.n_x = n_x
        self.n_t = n_t
        self.T = T
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.use_crank_nicolson = use_crank_nicolson
        self.fd_order = fd_order
        
        # Grid spacing
        self.Delta_x = 1.0 / n_x  # Domain [0,1]
        self.Delta_t = T / (n_t - 1)
        
        # Spatial grid (including boundaries)
        self.x_grid = np.linspace(0, 1, n_x + 1)
        
        # Time grid
        self.t_grid = np.linspace(0, T, n_t)
        
        # Interior points (excluding boundaries)
        self.x_interior = self.x_grid[1:-1]
        self.n_interior = len(self.x_interior)
        
    def build_diffusion_matrix(self, kappa_field: np.ndarray) -> csr_matrix:
        """
        Build the discrete diffusion operator ?(?(x)) using centered finite differences.
        
        For heterogeneous ?, at interior point i:
        (?(?u))_i H [?_{i+1/2}(u_{i+1}-u_i) - ?_{i-1/2}(u_i-u_{i-1})] / ?x?
        
        where ?_{i+1/2} = (?_i + ?_{i+1})/2 (harmonic mean could also be used)
        
        Args:
            kappa_field: Conductivity values at grid points (size n_x+1)
            
        Returns:
            Sparse matrix for diffusion operator (size n_interior ? n_interior)
        """
        n = self.n_interior
        dx2 = self.Delta_x ** 2
        
        # Compute interface conductivities (arithmetic mean)
        # ?_{i+1/2} for i=0,...,n_x-1
        kappa_plus = 0.5 * (kappa_field[1:-1] + kappa_field[2:])   # ?_{i+1/2}
        kappa_minus = 0.5 * (kappa_field[:-2] + kappa_field[1:-1])  # ?_{i-1/2}
        
        # Build tridiagonal matrix
        # Diagonal: -(?_{i+1/2} + ?_{i-1/2})/?x?
        diagonal = -(kappa_plus + kappa_minus) / dx2
        
        # Super-diagonal: ?_{i+1/2}/?x?
        super_diag = kappa_plus[:-1] / dx2
        
        # Sub-diagonal: ?_{i-1/2}/?x?
        sub_diag = kappa_minus[1:] / dx2
        
        # Construct sparse matrix
        A = diags([sub_diag, diagonal, super_diag], [-1, 0, 1], shape=(n, n), format='csr')
        
        return A
    
    def forward_solve(self, 
                     kappa: np.ndarray,
                     u_0: Optional[np.ndarray] = None,
                     f: Optional[Callable] = None,
                     sensor_positions: Optional[np.ndarray] = None,
                     sensor_times: Optional[np.ndarray] = None) -> Dict:
        """
        Solve the heat equation with given conductivity field.
        
        Args:
            kappa: Piecewise constant conductivity values (m segments)
                   or full field at grid points
            u_0: Initial condition (if None, use zero)
            f: Source term function f(x,t) (if None, use zero)
            sensor_positions: Sensor x-locations for output restriction
            sensor_times: Time points for output restriction
            
        Returns:
            Dictionary with:
                - 'full_solution': Complete solution u(x,t) if no sensors specified
                - 'sensor_values': Solution at sensor locations if specified
                - 'kappa_field': Interpolated conductivity field
        """
        # Expand kappa to full field if given as segments
        if len(kappa) < self.n_x + 1:
            kappa_field = self._expand_kappa_segments(kappa)
        else:
            kappa_field = kappa
            
        # Validate kappa bounds
        if np.any(kappa_field < self.kappa_min) or np.any(kappa_field > self.kappa_max):
            warnings.warn(f"? values outside bounds [{self.kappa_min}, {self.kappa_max}]")
            kappa_field = np.clip(kappa_field, self.kappa_min, self.kappa_max)
        
        # Initialize solution array
        u = np.zeros((self.n_interior, self.n_t))
        
        # Set initial condition
        if u_0 is None:
            u[:, 0] = 0.0  # Zero initial condition
        else:
            u[:, 0] = u_0[1:-1]  # Extract interior points
        
        # Build diffusion matrix
        A = self.build_diffusion_matrix(kappa_field)
        
        # Time stepping
        if self.use_crank_nicolson:
            self._crank_nicolson_timestepping(u, A, f)
        else:
            self._backward_euler_timestepping(u, A, f)
        
        # Extract solution at sensors if specified
        if sensor_positions is not None and sensor_times is not None:
            sensor_values = self._extract_sensor_values(u, sensor_positions, sensor_times)
            return {
                'sensor_values': sensor_values,
                'kappa_field': kappa_field,
                'full_solution': u
            }
        else:
            return {
                'full_solution': u,
                'kappa_field': kappa_field
            }
    
    def _expand_kappa_segments(self, kappa_segments: np.ndarray) -> np.ndarray:
        """
        Expand piecewise constant ? segments to full grid.
        
        ?(x) = ?_{r=1}^m ?_r ? 1_{I_r}(x)
        
        Args:
            kappa_segments: m segment values
            
        Returns:
            Conductivity at all grid points
        """
        m = len(kappa_segments)
        kappa_field = np.zeros(self.n_x + 1)
        
        # Equal-length segments
        segment_boundaries = np.linspace(0, 1, m + 1)
        
        for i, x in enumerate(self.x_grid):
            # Find which segment x belongs to
            segment_idx = np.searchsorted(segment_boundaries[1:], x)
            segment_idx = min(segment_idx, m - 1)  # Handle boundary case
            kappa_field[i] = kappa_segments[segment_idx]
        
        return kappa_field
    
    def _crank_nicolson_timestepping(self, u: np.ndarray, A: csr_matrix, 
                                    f: Optional[Callable]):
        """
        Crank-Nicolson time stepping: unconditionally stable, second-order in time.
        
        (I - ?t/2?A)u^{n+1} = (I + ?t/2?A)u^n + ?t?f^{n+1/2}
        """
        dt = self.Delta_t
        n = self.n_interior
        I = eye(n, format='csr')
        
        # Matrices for CN scheme
        A_implicit = I - 0.5 * dt * A  # Left-hand side
        A_explicit = I + 0.5 * dt * A  # Right-hand side
        
        # Time stepping loop
        for n_step in range(1, self.n_t):
            # Right-hand side
            rhs = A_explicit @ u[:, n_step - 1]
            
            # Add source term if provided
            if f is not None:
                # Evaluate at midpoint time
                t_mid = self.t_grid[n_step - 1] + 0.5 * dt
                f_mid = np.array([f(x, t_mid) for x in self.x_interior])
                rhs += dt * f_mid
            
            # Solve implicit system
            u[:, n_step] = spsolve(A_implicit, rhs)
    
    def _backward_euler_timestepping(self, u: np.ndarray, A: csr_matrix, 
                                    f: Optional[Callable]):
        """
        Backward Euler time stepping: unconditionally stable, first-order in time.
        
        (I - ?t?A)u^{n+1} = u^n + ?t?f^{n+1}
        """
        dt = self.Delta_t
        n = self.n_interior
        I = eye(n, format='csr')
        
        # Matrix for implicit scheme
        A_implicit = I - dt * A
        
        # Time stepping loop
        for n_step in range(1, self.n_t):
            # Right-hand side
            rhs = u[:, n_step - 1].copy()
            
            # Add source term if provided
            if f is not None:
                t_current = self.t_grid[n_step]
                f_current = np.array([f(x, t_current) for x in self.x_interior])
                rhs += dt * f_current
            
            # Solve implicit system
            u[:, n_step] = spsolve(A_implicit, rhs)
    
    def _extract_sensor_values(self, u: np.ndarray, 
                              sensor_positions: np.ndarray,
                              sensor_times: np.ndarray) -> np.ndarray:
        """
        Extract solution values at sensor locations and times.
        
        Args:
            u: Solution array (n_interior ? n_t)
            sensor_positions: x-coordinates of sensors
            sensor_times: Time indices for measurements
            
        Returns:
            Sensor measurements in lexicographic order (i,j)
        """
        n_sensors = len(sensor_positions)
        n_times = len(sensor_times)
        sensor_values = np.zeros(n_sensors * n_times)
        
        # Map sensor positions to grid indices
        sensor_indices = []
        for x_s in sensor_positions:
            # Find nearest interior grid point
            idx = np.argmin(np.abs(self.x_interior - x_s))
            sensor_indices.append(idx)
        
        # Map time points to grid indices
        time_indices = []
        for t in sensor_times:
            idx = np.argmin(np.abs(self.t_grid - t))
            time_indices.append(idx)
        
        # Extract values in lexicographic order
        counter = 0
        for i, x_idx in enumerate(sensor_indices):
            for j, t_idx in enumerate(time_indices):
                sensor_values[counter] = u[x_idx, t_idx]
                counter += 1
        
        return sensor_values
    
    def compute_discretization_error(self, kappa: np.ndarray,
                                    refined_solver: 'HeatEquationSolver',
                                    sensor_positions: np.ndarray,
                                    sensor_times: np.ndarray) -> float:
        """
        Compute discretization error ?_h by comparing with refined mesh.
        
        ?_h = sup_? |(y,F_h(?)) - (y,F_{h/2}(?))|
        
        Args:
            kappa: Conductivity field
            refined_solver: Solver with finer mesh (h/2)
            sensor_positions: Sensor locations
            sensor_times: Measurement times
            
        Returns:
            Discretization error estimate
        """
        # Solve on current mesh
        result_h = self.forward_solve(kappa, sensor_positions=sensor_positions,
                                     sensor_times=sensor_times)
        u_h = result_h['sensor_values']
        
        # Solve on refined mesh
        result_h2 = refined_solver.forward_solve(kappa, sensor_positions=sensor_positions,
                                                sensor_times=sensor_times)
        u_h2 = result_h2['sensor_values']
        
        # Compute difference
        error = np.max(np.abs(u_h - u_h2))
        
        return error
    
    def validate_solver(self, test_case: str = 'manufactured') -> Dict:
        """
        Unit test for solver accuracy using manufactured solution.
        
        Args:
            test_case: Type of validation test
            
        Returns:
            Dictionary with validation metrics
        """
        if test_case == 'manufactured':
            # Manufactured solution: u(x,t) = sin(?x)?exp(-??t)
            # This satisfies BC and heat equation with ?=1
            
            # Exact solution
            def u_exact(x, t):
                return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
            
            # Initial condition
            u_0 = np.array([u_exact(x, 0) for x in self.x_grid])
            
            # Constant conductivity
            kappa = np.ones(self.n_x + 1)
            
            # Solve
            result = self.forward_solve(kappa, u_0=u_0)
            u_numerical = result['full_solution']
            
            # Compare with exact at final time
            u_exact_final = np.array([u_exact(x, self.T) for x in self.x_interior])
            u_numerical_final = u_numerical[:, -1]
            
            # Compute errors
            l2_error = np.linalg.norm(u_numerical_final - u_exact_final) * np.sqrt(self.Delta_x)
            linf_error = np.max(np.abs(u_numerical_final - u_exact_final))
            
            # Check boundary conditions
            u_full = np.zeros((self.n_x + 1, self.n_t))
            u_full[1:-1, :] = u_numerical
            bc_error = np.max([np.abs(u_full[0, :]).max(), np.abs(u_full[-1, :]).max()])
            
            return {
                'l2_error': l2_error,
                'linf_error': linf_error,
                'bc_error': bc_error,
                'order_estimate': -np.log(linf_error) / np.log(self.Delta_x) if linf_error > 0 else np.inf
            }
        else:
            raise ValueError(f"Unknown test case: {test_case}")