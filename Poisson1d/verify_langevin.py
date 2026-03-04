"""
Script to verify Langevin dynamics for 1D Poisson.

Tests if running Langevin dynamics (gradient descent + noise) on the energy function
minimizes the energy and converges effectively to the true solution.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from energy import compute_energy, compute_energy_gradient
from langevin import langevin_step
from data import sample_coefficient_a, sample_forcing_f
from utils import solve_poisson_exact, get_grid_points, compute_relative_l2_error

def verify_langevin(
    n_grid=50,
    step_size=1e-4,
    noise_scale=1e-3,
    steps=1000,
    f_amplitude=1.0,
    deterministic=False,
    save_path='langevin_verification.png',
    force_type='grf'
):
    device = 'cpu'
    
    # 1. Setup problem
    # Coefficient a=1
    a = torch.ones(1, n_grid + 2, device=device)
    
    # Forcing f
    if force_type == 'sin_4pi':
        x = get_grid_points(n_grid, include_boundary=False).to(device)
        f = f_amplitude * torch.sin(4 * np.pi * x).unsqueeze(0)
    else:
        # Normalized GRF
        f = sample_forcing_f(n_grid, 1, method='grf', amplitude=f_amplitude, device=device)
    
    # Grid spacing
    h = 1.0 / (n_grid + 1)
    
    # 2. Exact solution for reference
    u_exact = solve_poisson_exact(a, f, h)
    
    # 3. Langevin Dynamics Process
    # Start from random noise
    u = torch.randn(1, n_grid, device=device)
    
    energies = []
    l2_errors = []
    
    actual_noise = 0.0 if deterministic else noise_scale
    
    print(f"Running {steps} Langevin steps...")
    print(f"Step size: {step_size}, Noise scale: {actual_noise}")
    
    for k in range(steps):
        # Record stats
        E = compute_energy(u, a, f, h).item()
        err = compute_relative_l2_error(u, u_exact, h).item()
        
        energies.append(E)
        l2_errors.append(err)
        
        # Step
        grad = compute_energy_gradient(u, a, f, h)
        noise = torch.randn_like(u)
        u = u - step_size * grad + actual_noise * noise
    
    # Final stats
    E_final = compute_energy(u, a, f, h).item()
    E_exact = compute_energy(u_exact, a, f, h).item()
    
    print(f"Final Energy: {E_final:.4f} (Exact: {E_exact:.4f})")
    print(f"Final L2 Error: {l2_errors[-1]:.4f}")
    
    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Trajectories
    axes[0].plot(energies, label='Langevin')
    axes[0].axhline(y=E_exact, color='r', linestyle='--', label='Exact Energy')
    axes[0].set_title('Energy vs Step')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Energy J(u)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(l2_errors)
    axes[1].set_title('Relative L2 Error vs Step')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('||u - u*|| / ||u*||')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Solution plot
    x_full = get_grid_points(n_grid, include_boundary=True).numpy()
    
    u_ex_plot = np.concatenate(([0], u_exact[0].numpy(), [0]))
    u_plot = np.concatenate(([0], u[0].numpy(), [0]))
    
    axes[2].plot(x_full, u_ex_plot, 'k-', linewidth=2, label='Exact')
    axes[2].plot(x_full, u_plot, 'r--', label='Langevin Sample')
    axes[2].set_title(f'Solution (Final Error: {l2_errors[-1]:.4f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--step_size', type=float, default=1e-4)
    parser.add_argument('--noise_scale', type=float, default=1e-3)
    parser.add_argument('--f_amp', type=float, default=10.0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--save_path', type=str, default='langevin_verification.png')
    parser.add_argument('--force_type', type=str, default='grf', choices=['grf', 'sin_4pi'])
    parser.add_argument('--seed', type=int, default=26, help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    verify_langevin(
        steps=args.steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale,
        f_amplitude=args.f_amp,
        deterministic=args.deterministic,
        save_path=args.save_path,
        force_type=args.force_type
    )
