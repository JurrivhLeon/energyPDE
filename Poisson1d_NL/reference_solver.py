
import numpy as np
import torch
import scipy.integrate
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add current directory to path to import data and energy modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import sample_source_s
from energy import compute_energy_gradient, build_laplacian_matrix

def solve_ginzburg_landau_bvp(s_values, x_grid, tol=1e-6, max_nodes=10000):
    """
    Solve the 1D Ginzburg-Landau BVP:
        -u'' + u^3 - u = s(x)
        u(0) = u(1) = 0
        
    Args:
        s_values: Discrete values of source term s on x_grid.
        x_grid: Grid points corresponding to s_values (must be sorted, [0, 1]).
        tol: Tolerance for solve_bvp.
        max_nodes: Maximum number of mesh nodes.
        
    Returns:
        sol_u: Solution u evaluate on x_grid.
        res: The solver result object.
    """
    # 1. Create continuous interpolation of s(x)
    s_spline = CubicSpline(x_grid, s_values)
    
    def fun(x, y):
        """
        ODE system:
        y[0] = u
        y[1] = u'
        
        u' = y[1]
        u'' = u^3 - u - s(x)
           => y[1]' = y[0]^3 - y[0] - s(x)
        """
        u = y[0]
        v = y[1] # u'
        
        du_dx = v
        dv_dx = u**3 - u - s_spline(x)
        
        return np.vstack((du_dx, dv_dx))

    def bc(ya, yb):
        """
        Boundary conditions:
        u(0) = 0  => ya[0] = 0
        u(1) = 0  => yb[0] = 0
        """
        return np.array([ya[0], yb[0]])

    # Initial mesh
    x_init = x_grid
    
    # Initial guess: u(x) = 0, u'(x) = 0
    # y shape: (2, n_points)
    y_init = np.zeros((2, x_init.size))
    
    # Solve BVP
    res = scipy.integrate.solve_bvp(fun, bc, x_init, y_init, tol=tol, max_nodes=max_nodes)
    
    if not res.success:
        print(f"Warning: BVP solver failed to converge: {res.message}")
        
    # Evaluate solution on the original grid
    sol_y = res.sol(x_grid)
    u_sol = sol_y[0]
    
    return u_sol, res

def main():
    parser = argparse.ArgumentParser(description="Ginzburg-Landau Reference Solver")
    parser.add_argument('--n', type=int, default=100, help="Number of interior grid points")
    parser.add_argument('--tol', type=float, default=1e-6, help="Solver tolerance")
    parser.add_argument('--plot', action='store_true', help="Plot method solution")
    parser.add_argument('--method', type=str, default='grf', choices=['grf', 'matern', 'sinusoidal', 'constant'], help="Source generation method")
    args = parser.parse_args()
    
    # Grid setup
    n_interior = args.n
    h = 1.0 / (n_interior + 1)
    x_interior = np.linspace(h, 1-h, n_interior)
    x_full = np.concatenate(([0], x_interior, [1]))
    
    # Sample source term (using data.py)
    print(f"Generating source term using method: {args.method}...")
    # Use a reasonable amplitude; GRF/Matern can vary.
    s_torch = sample_source_s(n_interior, n_samples=1, method=args.method, amplitude=10.0)
    s_interior = s_torch.numpy().flatten()
    
    # Pad for interpolation
    s_full = np.concatenate(([s_interior[0]], s_interior, [s_interior[-1]]))
    
    print("Solving BVP...")
    u_full, res = solve_ginzburg_landau_bvp(s_full, x_full, tol=args.tol)
    
    u_interior = u_full[1:-1]
    
    # Check discrete residual
    K = build_laplacian_matrix(n_interior, h)
    u_torch = torch.from_numpy(u_interior).float()
    s_torch_check = torch.from_numpy(s_interior).float()
    
    grad = compute_energy_gradient(u_torch, s_torch_check, h, K=K)
    residual_norm = grad.norm().item()
    
    print(f"Solver success: {res.success}")
    print(f"Number of mesh nodes: {res.x.size}")
    print(f"Discrete Energy Gradient Norm (Residual): {residual_norm:.2e}")
    
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        ax1.plot(x_full, u_full, 'b-', label='Solution u(x)', linewidth=2)
        ax1.set_ylabel('u(x)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title(f'Ginzburg-Landau Solution (Method: {args.method})')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x_full, s_full, 'r--', label='Source s(x)', linewidth=1.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('s(x)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'reference_solution_{args.method}.png'
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to {filename}")

if __name__ == "__main__":
    main()
