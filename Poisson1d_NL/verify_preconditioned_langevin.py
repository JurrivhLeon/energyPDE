"""
Script to verify preconditioned Langevin dynamics for the Nonlinear Poisson equation.

Tests if running preconditioned Langevin dynamics on the energy function 
minimizes the energy and converges effectively, comparing it with standard Langevin.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

from preconditioned_langevin import MaternPreconditioner, preconditioned_langevin_step, dst_type1
from langevin import langevin_step
from energy import compute_energy, compute_energy_gradient, build_laplacian_matrix
from data import sample_source_s
from utils import get_grid_points, compute_relative_l2_error
from reference_solver import solve_ginzburg_landau_bvp



def verify_preconditioned_langevin(
    n_grid=64,
    kappa=1.0,
    alpha=2.0,
    step_size_std=1e-4,
    step_size_pre=1e-3,
    noise_scale=1e-3,
    steps=1000,
    force_type='sinusoidal',
    f_amplitude=10.0,
    save_path='preconditioned_verification.png',
    seed=42,
    deterministic=False,
    scaling='max',
    grad_clip=None,
    compute_ref=True,
    nu=2.5,
    freq=1.0,
    length_scale=0.2
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode_str = "Deterministic" if deterministic else "Stochastic"
    print(f"Running verification on {device} ({mode_str}, scaling={scaling})")
    
    # 1. Setup problem
    h = 1.0 / (n_grid + 1)
    
    # Source s
    if force_type == 'sinusoidal':
        x_int = get_grid_points(n_grid, include_boundary=False, device=device)
        s = f_amplitude * torch.sin(2 * np.pi * freq * x_int).unsqueeze(0)
    elif force_type == 'grf':
        s = sample_source_s(n_grid, 1, method='grf', amplitude=f_amplitude, length_scale=length_scale, device=device)
    elif force_type == 'matern':
        s = sample_source_s(n_grid, 1, method='matern', amplitude=f_amplitude, length_scale=length_scale, nu=nu, device=device)
    else:
        s = f_amplitude * torch.ones(1, n_grid, device=device)
        
    s = s.to(device)
        
    # Precompute Stiffness matrix
    K = build_laplacian_matrix(n_grid, h, device=device)
    
    # 2. Reference Solution
    u0_ref = torch.zeros(1, n_grid, device=device) 
    
    if compute_ref:
        x_interior = get_grid_points(n_grid, include_boundary=False, device='cpu').numpy()
        x_full = get_grid_points(n_grid, include_boundary=True, device='cpu').numpy()
        
        s_cpu = s[0].cpu().numpy()
        s_full = np.concatenate(([s_cpu[0]], s_cpu, [s_cpu[-1]]))
        
        u_bvp, res = solve_ginzburg_landau_bvp(s_full, x_full, tol=1e-8)
        
        if not res.success:
            print(f"BVP Solver Warning: {res.message}")
            
        u_interior = u_bvp[1:-1]
        u_exact = torch.from_numpy(u_interior).float().to(device).unsqueeze(0)
        
        E_exact = compute_energy(u_exact, s, h, K=K).item()
    else:
        u_exact = torch.zeros_like(u0_ref)
        E_exact = 0.0
    
    # 3. Initialize
    if deterministic:
        u0_std = torch.zeros(1, n_grid, device=device)
        u0_pre = torch.zeros(1, n_grid, device=device)
    else:
        u0_std = torch.randn(1, n_grid, device=device)
        
        precond_init = MaternPreconditioner(
            n_grid, kappa=kappa, alpha=alpha, 
            normalize=True, mode='matern', device=device, scaling=scaling
        )
        u0_pre = precond_init.sample_noise((1, n_grid))
    
    # Preconditioner for dynamics
    precond = MaternPreconditioner(
        n_grid, kappa=kappa, alpha=alpha, 
        normalize=True, mode='matern', device=device, scaling=scaling
    )
    
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
    u = results['std']['u'].clone()
    start_time = time.time()
    
    for k in range(steps):
        E = compute_energy(u, s, h, K=K).item()
        if compute_ref:
            err = compute_relative_l2_error(u, u_exact, h).item()
        else:
            err = 0.0
        results['std']['energy'].append(E)
        results['std']['l2'].append(err)
        
        grad = compute_energy_gradient(u, s, h, K=K)
        
        if deterministic:
            noise = torch.zeros_like(u)
        else:
            noise = torch.randn_like(u)
            
        u = u - step_size_std * grad + actual_noise * noise
        
    results['std']['u'] = u
    print(f"Standard done in {time.time() - start_time:.2f}s")

    # Preconditioned Langevin / GD
    u = results['pre']['u']
    start_time = time.time()
    for k in range(steps):
        E = compute_energy(u, s, h, K=K).item()
        if compute_ref:
            err = compute_relative_l2_error(u, u_exact, h).item()
        else:
            err = 0.0
            
        results['pre']['energy'].append(E)
        results['pre']['l2'].append(err)
        
        u = preconditioned_langevin_step(
            u, s, h, step_size_pre, actual_noise, precond, K=K,
            grad_clip=grad_clip
        )
    results['pre']['u'] = u
    print(f"Preconditioned done in {time.time() - start_time:.2f}s")
    
    print(f"\nResults:")
    if compute_ref:
        print(f"  Ref Energy: {E_exact:.4f}")
    print(f"  Standard: Final E={results['std']['energy'][-1]:.4f}")
    print(f"  Precond:  Final E={results['pre']['energy'][-1]:.4f}")
    
    # 5. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Energy
    ax = axes[0, 0]
    ax.plot(results['std']['energy'], label='Standard')
    ax.plot(results['pre']['energy'], label='Precond')
    if compute_ref:
        ax.axhline(y=E_exact, color='k', linestyle='--', label='Reference')
    ax.set_title('Energy vs Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy J(u)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. L2 Error
    if compute_ref:
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
    if compute_ref:
        u_ex_plot = np.concatenate(([0], u_exact[0].cpu().numpy(), [0]))
        ax.plot(x_full, u_ex_plot, 'k-', linewidth=2, label='Ref')
        
    u_std_plot = np.concatenate(([0], results['std']['u'][0].cpu().numpy(), [0]))
    u_pre_plot = np.concatenate(([0], results['pre']['u'][0].cpu().numpy(), [0]))
    
    ax.plot(x_full, u_std_plot, 'r--', label=f'Standard')
    ax.plot(x_full, u_pre_plot, 'b--', label=f'Precond')
    ax.set_title('Final Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Matern Noise Samples
    ax = axes[1, 0]
    x_int = get_grid_points(n_grid, include_boundary=False).cpu().numpy()
    for i in range(5):
        noise = precond.sample_noise((n_grid,)).cpu().numpy()
        ax.plot(x_int, noise, alpha=0.8, label=f'Sample {i+1}')
    ax.set_title(f'Matérn Noise (κ={kappa}, α={alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Spectrum
    ax = axes[1, 1]
    modes = np.arange(1, n_grid + 1)
    u_std_hat = dst_type1(results['std']['u'][0]).abs().cpu().numpy()
    u_pre_hat = dst_type1(results['pre']['u'][0]).abs().cpu().numpy()
    if compute_ref:
        u_ex_hat = dst_type1(u_exact[0]).abs().cpu().numpy()
        ax.semilogy(modes, u_ex_hat, 'k-', linewidth=2, label='Ref')
        
    ax.semilogy(modes, u_std_hat, 'r--', label='Standard')
    ax.semilogy(modes, u_pre_hat, 'b--', label='Precond')
    ax.set_title('Spectral Content (DST)')
    ax.set_xlabel('Mode m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Source Term
    ax = axes[1, 2]
    s_plot = s[0].cpu().numpy()
    ax.plot(x_int, s_plot, 'g-', label='Source s(x)')
    ax.set_title('Source Term')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

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
    parser.add_argument('--force_type', type=str, default='matern', choices=['grf', 'sinusoidal', 'constant', 'matern'])
    parser.add_argument('--f_amp', type=float, default=20.0)
    parser.add_argument('--freq', type=float, default=3.0, help="Frequency for sinusoidal source")
    parser.add_argument('--length_scale', type=float, default=0.2, help="Length scale for GRF/Matern source")
    parser.add_argument('--nu', type=float, default=1.5, help="Smoothness for Matern source (0.5, 1.5, 2.5)")
    parser.add_argument('--seed', type=int, default=1344)
    parser.add_argument('--save_path', type=str, default='verification_results_nl.png')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--deterministic', action='store_true',
                        help='Run deterministic PGD (noise_scale=0)')
    parser.add_argument('--scaling', type=str, default='max', choices=['max', 'trace'])
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--no_ref', action='store_true', help="Skip computing reference solution")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
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
        grad_clip=args.grad_clip,
        compute_ref=not args.no_ref,
        nu=args.nu,
        freq=args.freq,
        length_scale=args.length_scale
    )
