"""
Script to verify preconditioned Langevin dynamics.

Tests if running preconditioned Langevin dynamics on the energy function 
minimizes the energy and converges effectively, comparing it with standard Langevin.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from preconditioned_langevin import MaternPreconditioner, preconditioned_langevin_step, dst_type1
from langevin import langevin_step
from energy import compute_energy, compute_energy_gradient
from data import sample_forcing_f
from utils import solve_poisson_exact, get_grid_points, compute_relative_l2_error

def verify_preconditioned_langevin(
    n_grid=64,
    kappa=1.0,
    alpha=2.0,
    step_size_std=1e-4,
    step_size_pre=1e-3,
    noise_scale=1e-3,
    steps=1000,
    force_type='grf',
    f_amplitude=10.0,
    save_path='preconditioned_verification.png',
    seed=42,
    deterministic=False,
    scaling='max',
    grad_clip=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode_str = "Deterministic" if deterministic else "Stochastic"
    print(f"Running verification on {device} ({mode_str}, scaling={scaling})")
    if grad_clip:
        print(f"Gradient clipping enabled: {grad_clip}")
    
    # 1. Setup problem
    # Coefficient a=1
    a = torch.ones(1, n_grid + 2, device=device)
    
    # Forcing f
    if force_type == 'grf':
        f = sample_forcing_f(n_grid, 1, method='grf', amplitude=f_amplitude, device=device)
    elif force_type == 'sinusoidal':
        f = sample_forcing_f(n_grid, 1, method='sinusoidal', amplitude=f_amplitude, device=device)
    elif force_type == 'sin_4pi':
        x = get_grid_points(n_grid, include_boundary=False).to(device)
        f = f_amplitude * torch.sin(4 * np.pi * x).unsqueeze(0)
    elif force_type == 'mixed':
        # Randomly pick grf or sinusoidal
        method = 'grf' if torch.rand(1).item() < 0.5 else 'sinusoidal'
        f = sample_forcing_f(n_grid, 1, method=method, amplitude=f_amplitude, device=device)
    else:
        f = sample_forcing_f(n_grid, 1, method='grf', amplitude=f_amplitude, device=device)
    
    # Grid spacing
    h = 1.0 / (n_grid + 1)
    
    # 2. Exact solution for reference
    u_exact = solve_poisson_exact(a, f, h)
    
    # 3. Initialize
    if deterministic:
        u0_std = torch.zeros(1, n_grid, device=device)
    else:
        u0_std = torch.randn(1, n_grid, device=device)
    
    # Preconditioner
    precond = MaternPreconditioner(
        n_grid, kappa=kappa, alpha=alpha, 
        normalize=True, mode='matern', device=device, scaling=scaling
    )
    
    if deterministic:
        u0_pre = torch.zeros(1, n_grid, device=device)
    else:
        u0_pre = precond.sample_noise((1, n_grid))
    
    # Actual noise scale (0 if deterministic)
    actual_noise = 0.0 if deterministic else noise_scale
    
    # Storage
    label_std = f'Std {"GD" if deterministic else "Langevin"} (ss={step_size_std})'
    label_pre = f'Pre {"GD" if deterministic else "Langevin"} (ss={step_size_pre})'
    
    results = {
        'std': {'u': u0_std.clone(), 'energy': [], 'l2': [], 'label': label_std},
        'pre': {'u': u0_pre.clone(), 'energy': [], 'l2': [], 'label': label_pre}
    }
    
    print(f"\nRunning {steps} steps...")
    print(f"Standard:       step_size={step_size_std}")
    print(f"Preconditioned: step_size={step_size_pre}, kappa={kappa}, alpha={alpha}")
    print(f"Noise scale:    {actual_noise}")
    
    # 4. Run Dynamics
    
    # Standard Langevin / GD
    u = results['std']['u']
    for k in range(steps):
        E = compute_energy(u, a, f, h).item()
        err = compute_relative_l2_error(u, u_exact, h).item()
        results['std']['energy'].append(E)
        results['std']['l2'].append(err)
        
        grad = compute_energy_gradient(u, a, f, h)
        noise = torch.randn_like(u)
        u = u - step_size_std * grad + actual_noise * noise
    results['std']['u'] = u
    
    # Preconditioned Langevin / GD
    u = results['pre']['u']
    for k in range(steps):
        E = compute_energy(u, a, f, h).item()
        err = compute_relative_l2_error(u, u_exact, h).item()
        results['pre']['energy'].append(E)
        results['pre']['l2'].append(err)
        
        u = preconditioned_langevin_step(
            u, a, f, h, step_size_pre, actual_noise, precond,
            grad_clip=grad_clip
        )
    results['pre']['u'] = u
    
    # Exact Energy
    E_exact = compute_energy(u_exact, a, f, h).item()
    
    print(f"\nResults:")
    print(f"  Exact Energy: {E_exact:.4f}")
    print(f"  Standard: Final L2={results['std']['l2'][-1]:.4f}, E={results['std']['energy'][-1]:.4f}")
    print(f"  Precond:  Final L2={results['pre']['l2'][-1]:.4f}, E={results['pre']['energy'][-1]:.4f}")
    
    # 5. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Energy
    ax = axes[0, 0]
    ax.plot(results['std']['energy'], label='Standard')
    ax.plot(results['pre']['energy'], label='Precond')
    ax.axhline(y=E_exact, color='k', linestyle='--', label='Exact')
    ax.set_title('Energy vs Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy J(u)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. L2 Error
    ax = axes[0, 1]
    ax.plot(results['std']['l2'], label='Standard')
    ax.plot(results['pre']['l2'], label='Precond')
    ax.set_title('Relative L2 Error vs Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('||u - u*|| / ||u*||')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Solution
    ax = axes[0, 2]
    x_full = get_grid_points(n_grid, include_boundary=True).cpu().numpy()
    u_ex_plot = np.concatenate(([0], u_exact[0].cpu().numpy(), [0]))
    u_std_plot = np.concatenate(([0], results['std']['u'][0].cpu().numpy(), [0]))
    u_pre_plot = np.concatenate(([0], results['pre']['u'][0].cpu().numpy(), [0]))
    
    ax.plot(x_full, u_ex_plot, 'k-', linewidth=2, label='Exact')
    ax.plot(x_full, u_std_plot, 'r--', label=f'Standard')
    ax.plot(x_full, u_pre_plot, 'b--', label=f'Precond')
    ax.set_title('Final Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Matérn Noise Samples
    ax = axes[1, 0]
    x_int = get_grid_points(n_grid, include_boundary=False).cpu().numpy()
    for i in range(10):
        noise = precond.sample_noise((n_grid,)).cpu().numpy()
        ax.plot(x_int, noise, alpha=0.8, label=f'Sample {i+1}')
    ax.set_title(f'Matérn Noise (κ={kappa}, α={alpha})')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Spectrum
    ax = axes[1, 1]
    modes = np.arange(1, n_grid + 1)
    u_std_hat = dst_type1(results['std']['u'][0]).abs().cpu().numpy()
    u_pre_hat = dst_type1(results['pre']['u'][0]).abs().cpu().numpy()
    u_ex_hat = dst_type1(u_exact[0]).abs().cpu().numpy()
    
    ax.semilogy(modes, u_ex_hat, 'k-', linewidth=2, label='Exact')
    ax.semilogy(modes, u_std_hat, 'r--', label='Standard')
    ax.semilogy(modes, u_pre_hat, 'b--', label='Precond')
    ax.set_title('Spectral Content (DST)')
    ax.set_xlabel('Mode m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Error Distribution
    ax = axes[1, 2]
    err_std = (results['std']['u'] - u_exact)[0].cpu().numpy()
    err_pre = (results['pre']['u'] - u_exact)[0].cpu().numpy()
    
    ax.plot(x_int, err_std, 'r-', label='Standard Error')
    ax.plot(x_int, err_pre, 'b-', label='Precond Error')
    ax.set_title('Error Distribution')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

def visualize_matern_noise(n_grid, kappa, alpha, device='cpu', scaling='max', n_samples=10, save_path='matern_noise.png'):
    """Visualize samples from the Matern noise process."""
    precond = MaternPreconditioner(
        n_grid, kappa=kappa, alpha=alpha, 
        normalize=True, mode='matern', device=device, scaling=scaling
    )
    
    x = get_grid_points(n_grid, include_boundary=True).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    
    for i in range(n_samples):
        noise = precond.sample_noise((n_grid,)).cpu()
        noise_plot = np.concatenate(([0], noise.numpy(), [0]))
        plt.plot(x, noise_plot, linewidth=2, label=f'Matern Sample {i+1}')
        
    plt.title(f'Matern Noise Process (κ={kappa}, α={alpha})')
    plt.xlabel('x')
    plt.ylabel('ξ(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved noise visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_grid', type=int, default=64)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--step_size_std', type=float, default=1e-3,
                        help='Step size for standard gradient/Langevin')
    parser.add_argument('--step_size_pre', type=float, default=1,
                        help='Step size for preconditioned gradient/Langevin')
    parser.add_argument('--noise_scale', type=float, default=1e-4,
                        help='Noise magnitude (overridden to 0 if --deterministic)')
    parser.add_argument('--force_type', type=str, default='mixed', choices=['grf', 'sinusoidal', 'sin_4pi', 'mixed'])
    parser.add_argument('--f_amp', type=float, default=500.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='verification_results.png')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--deterministic', action='store_true',
                        help='Run deterministic PGD (noise_scale=0)')
    parser.add_argument('--scaling', type=str, default='max', choices=['max', 'trace'],
                        help='Preconditioner scaling')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Gradient clipping value (optional)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Run noise visualization first
    visualize_matern_noise(args.n_grid, args.kappa, args.alpha, device=device, scaling=args.scaling)
    
    verify_preconditioned_langevin(
        n_grid=args.n_grid,
        kappa=args.kappa,
        alpha=args.alpha,
        step_size_std=args.step_size_std,
        step_size_pre=args.step_size_pre,
        noise_scale=args.noise_scale,
        steps=args.steps,
        f_amplitude=args.f_amp,
        force_type=args.force_type,
        save_path=args.save_path,
        seed=args.seed,
        deterministic=args.deterministic,
        scaling=args.scaling,
        grad_clip=args.grad_clip
    )
