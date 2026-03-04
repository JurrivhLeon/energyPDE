"""
Script to verify preconditioned Langevin dynamics for the 1D Burgers equation.

Tests if running preconditioned Langevin dynamics on the residual energy
minimizes the energy and converges to the reference backward Euler solution,
comparing it with standard (unpreconditioned) Langevin.

Uses two tunable parameters:
    step_size: gradient step size
    noise_scale: noise magnitude (0 = deterministic gradient descent)

Usage examples:
    # Deterministic gradient descent comparison
    python verify_preconditioned_langevin.py --deterministic --steps 2000

    # Stochastic Langevin comparison
    python verify_preconditioned_langevin.py --steps 3000

    # Tune preconditioner parameters
    python verify_preconditioned_langevin.py --kappa 2.0 --alpha 1.5 --step_size_pre 5e-4

    # Larger viscosity (easier problem)
    python verify_preconditioned_langevin.py --viscosity 0.1

    # Different IC type
    python verify_preconditioned_langevin.py --ic_type sinusoidal --ic_amp 2.0
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

from preconditioned_langevin import FourierPreconditioner, preconditioned_langevin_step
from energy import compute_energy, compute_energy_gradient
from data import sample_initial_condition
from reference_solver import solve_burgers_reference
from utils import get_grid_points, compute_relative_l2_error


def verify_preconditioned_langevin(
    n_grid=64,
    L=1.0,
    viscosity=0.01,
    dt=0.01,
    kappa=1.0,
    alpha=2.0,
    step_size_std=1e-4,
    step_size_pre=1e-3,
    noise_scale=0.0,
    steps=2000,
    ic_type='sinusoidal',
    ic_amp=1.0,
    ic_length_scale=0.3,
    save_path='verification_results_burgers.png',
    seed=42,
    deterministic=False,
    scaling='max',
    grad_clip=None,
    compute_ref=True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual_noise = 0.0 if deterministic else noise_scale
    mode_str = "Deterministic" if deterministic else f"Stochastic (noise={actual_noise})"
    print(f"Running verification on {device} ({mode_str}, scaling={scaling})")

    # --- 1. Setup problem ---
    h = L / n_grid
    x = get_grid_points(n_grid, L, device=device)

    # Generate initial condition u^n
    torch.manual_seed(seed)
    if ic_type == 'sinusoidal':
        # Must be periodic on [0, L]: sin(2*pi*x/L) satisfies f(0) = f(L) = 0
        u_curr = ic_amp * torch.sin(2 * np.pi * x / L).unsqueeze(0).to(device)
    elif ic_type == 'multi_mode':
        # Sum of periodic modes: k * 2*pi*x/L for integer k
        u_curr = ic_amp * (
            torch.sin(2 * np.pi * x / L)
            + 0.5 * torch.sin(6 * np.pi * x / L)
            + 0.3 * torch.cos(4 * np.pi * x / L)
        ).unsqueeze(0).to(device)
    else:
        u_curr = sample_initial_condition(
            n_grid, 1, method=ic_type, amplitude=ic_amp,
            length_scale=ic_length_scale, L=L, device=device
        )

    print(f"IC type: {ic_type}, amplitude: {ic_amp}")
    print(f"Physics: nu={viscosity}, dt={dt}, L={L:.4f}")

    # --- 2. Reference solution ---
    if compute_ref:
        print("Computing reference solution...")
        u_exact = solve_burgers_reference(u_curr, viscosity, dt, L)
        E_exact = compute_energy(u_exact, u_curr, viscosity, dt, L).item()
        print(f"Reference energy: {E_exact:.2e}")
    else:
        u_exact = torch.zeros_like(u_curr)
        E_exact = 0.0

    # --- 3. Initialize proposals ---
    torch.manual_seed(seed + 1)

    if deterministic:
        u0_std = u_curr.clone()
        u0_pre = u_curr.clone()
    else:
        u0_std = u_curr + 0.5 * torch.randn(1, n_grid, device=device)
        u0_pre = u0_std.clone()

    # Preconditioner
    precond = FourierPreconditioner(
        n_grid, kappa=kappa, alpha=alpha, L=L,
        normalize=True, scaling=scaling, device=device
    )

    # --- 4. Storage ---
    label_std = f'Std {"GD" if deterministic else "Langevin"} (s={step_size_std})'
    label_pre = f'Pre {"GD" if deterministic else "Langevin"} (s={step_size_pre})'

    results = {
        'std': {'u': u0_std.clone(), 'energy': [], 'l2': [], 'label': label_std},
        'pre': {'u': u0_pre.clone(), 'energy': [], 'l2': [], 'label': label_pre}
    }

    print(f"\nRunning {steps} steps...")
    print(f"Standard:       step_size={step_size_std}")
    print(f"Preconditioned: step_size={step_size_pre}, kappa={kappa}, alpha={alpha}")
    print(f"Noise scale:    {actual_noise}")

    # --- 5. Standard Langevin / GD ---
    u = results['std']['u'].clone()
    start_time = time.time()

    for k in range(steps):
        E = compute_energy(u, u_curr, viscosity, dt, L).item()
        if compute_ref:
            err = compute_relative_l2_error(u, u_exact, h).item()
        else:
            err = 0.0

        results['std']['energy'].append(E)
        results['std']['l2'].append(err)

        grad = compute_energy_gradient(u, u_curr, viscosity, dt, L)

        if grad_clip is not None:
            grad_norm = grad.norm(dim=-1, keepdim=True)
            grad = grad * torch.clamp(grad_clip / (grad_norm + 1e-8), max=1.0)

        noise = torch.randn_like(u) if not deterministic else torch.zeros_like(u)
        u = u - step_size_std * grad + actual_noise * noise

    results['std']['u'] = u
    print(f"Standard done in {time.time() - start_time:.2f}s, "
          f"final E={results['std']['energy'][-1]:.4e}")

    # --- 6. Preconditioned Langevin / GD ---
    u = results['pre']['u'].clone()
    start_time = time.time()

    for k in range(steps):
        E = compute_energy(u, u_curr, viscosity, dt, L).item()
        if compute_ref:
            err = compute_relative_l2_error(u, u_exact, h).item()
        else:
            err = 0.0

        results['pre']['energy'].append(E)
        results['pre']['l2'].append(err)

        u = preconditioned_langevin_step(
            u, u_curr, viscosity, dt, L,
            step_size_pre, actual_noise, precond,
            grad_clip=grad_clip
        )

    results['pre']['u'] = u
    print(f"Preconditioned done in {time.time() - start_time:.2f}s, "
          f"final E={results['pre']['energy'][-1]:.4e}")

    # --- 7. Summary ---
    print(f"\nResults (nu={viscosity}, dt={dt}):")
    if compute_ref:
        print(f"  Ref Energy:    {E_exact:.4e}")
    print(f"  Standard:  Final E={results['std']['energy'][-1]:.4e}, "
          f"L2={results['std']['l2'][-1]:.6f}")
    print(f"  Precond:   Final E={results['pre']['energy'][-1]:.4e}, "
          f"L2={results['pre']['l2'][-1]:.6f}")

    # --- 8. Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Burger1d Verification: ν={viscosity}, Δt={dt}, '
        f'n={n_grid}, {mode_str}',
        fontsize=14
    )

    # (0,0) Energy trajectory
    ax = axes[0, 0]
    ax.plot(results['std']['energy'], label='Standard', alpha=0.8)
    ax.plot(results['pre']['energy'], label='Preconditioned', alpha=0.8)
    if compute_ref:
        ax.axhline(y=E_exact, color='k', linestyle='--', label='Reference', alpha=0.7)
    ax.set_title('Energy vs Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy H(u)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) L2 Error
    ax = axes[0, 1]
    if compute_ref:
        ax.plot(results['std']['l2'], label='Standard', alpha=0.8)
        ax.plot(results['pre']['l2'], label='Preconditioned', alpha=0.8)
        ax.set_title('Relative L2 Error vs Step')
        ax.set_xlabel('Step')
        ax.set_ylabel('||u - u*|| / ||u*||')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Reference skipped', ha='center', va='center',
                transform=ax.transAxes)

    # (0,2) Final solution vs reference
    ax = axes[0, 2]
    x_np = x.cpu().numpy()

    ax.plot(x_np, u_curr[0].cpu().numpy(), 'k:', linewidth=1.5,
            label='u_curr (input)', alpha=0.6)
    if compute_ref:
        ax.plot(x_np, u_exact[0].cpu().numpy(), 'k-', linewidth=2, label='Reference')
    ax.plot(x_np, results['std']['u'][0].cpu().numpy(), 'r--', label='Standard')
    ax.plot(x_np, results['pre']['u'][0].cpu().numpy(), 'b--', label='Preconditioned')
    ax.set_title('Final Solution u^{n+1}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Preconditioner noise samples
    ax = axes[1, 0]
    for i in range(5):
        noise = precond.sample_noise((1, n_grid))[0].cpu().numpy()
        ax.plot(x_np, noise, alpha=0.7, label=f'Sample {i+1}')
    ax.set_title(f'Matérn Noise (κ={kappa}, α={alpha})')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Spectral content (FFT)
    ax = axes[1, 1]
    modes = np.arange(n_grid // 2 + 1)

    u_std_hat = np.abs(np.fft.rfft(results['std']['u'][0].cpu().numpy()))
    u_pre_hat = np.abs(np.fft.rfft(results['pre']['u'][0].cpu().numpy()))
    if compute_ref:
        u_ex_hat = np.abs(np.fft.rfft(u_exact[0].cpu().numpy()))
        ax.semilogy(modes[1:], u_ex_hat[1:], 'k-', linewidth=2, label='Reference')

    ax.semilogy(modes[1:], u_std_hat[1:], 'r--', alpha=0.8, label='Standard')
    ax.semilogy(modes[1:], u_pre_hat[1:], 'b--', alpha=0.8, label='Preconditioned')
    ax.set_title('Spectral Content |û_k| (FFT)')
    ax.set_xlabel('Mode k')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,2) Initial condition
    ax = axes[1, 2]
    ax.plot(x_np, u_curr[0].cpu().numpy(), 'g-', linewidth=2, label='u_curr (IC)')
    ax.set_title('Initial Condition u^n')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Verify preconditioned Langevin for Burgers equation'
    )

    # Grid and physics
    parser.add_argument('--n_grid', type=int, default=128)
    parser.add_argument('--L', type=float, default=1.0,
                        help='Domain length')
    parser.add_argument('--viscosity', type=float, default=0.1,
                        help='Viscosity nu')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step size')

    # Langevin
    parser.add_argument('--steps', type=int, default=2000,
                        help='Number of Langevin / GD steps')
    parser.add_argument('--step_size_std', type=float, default=1e-4,
                        help='Step size for standard gradient/Langevin')
    parser.add_argument('--step_size_pre', type=float, default=1e-4,
                        help='Step size for preconditioned gradient/Langevin')
    parser.add_argument('--noise_scale', type=float, default=1e-8,
                        help='Noise magnitude (overridden to 0 if --deterministic)')

    # Preconditioner
    parser.add_argument('--kappa', type=float, default=1.0,
                        help='Matérn kappa (inverse length scale)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Matérn alpha (smoothness exponent)')
    parser.add_argument('--scaling', type=str, default='max',
                        choices=['max'],
                        help='Eigenvalue normalization')

    # Initial condition
    parser.add_argument('--ic_type', type=str, default='grf',
                        choices=['sinusoidal', 'multi_mode', 'grf', 'mixed'],
                        help='Type of initial condition u^n')
    parser.add_argument('--ic_amp', type=float, default=1.0,
                        help='IC amplitude')
    parser.add_argument('--ic_length_scale', type=float, default=0.3,
                        help='GRF IC length scale')

    # Mode
    parser.add_argument('--deterministic', action='store_true',
                        help='Run deterministic gradient descent (noise_scale=0)')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Gradient clipping value')
    parser.add_argument('--no_ref', action='store_true',
                        help='Skip computing reference solution')

    # Output
    parser.add_argument('--seed', type=int, default=1344)
    parser.add_argument('--save_path', type=str,
                        default='verification_results_burgers.png')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    verify_preconditioned_langevin(
        n_grid=args.n_grid,
        L=args.L,
        viscosity=args.viscosity,
        dt=args.dt,
        kappa=args.kappa,
        alpha=args.alpha,
        step_size_std=args.step_size_std,
        step_size_pre=args.step_size_pre,
        noise_scale=args.noise_scale,
        steps=args.steps,
        ic_type=args.ic_type,
        ic_amp=args.ic_amp,
        ic_length_scale=args.ic_length_scale,
        save_path=args.save_path,
        seed=args.seed,
        deterministic=args.deterministic,
        scaling=args.scaling,
        grad_clip=args.grad_clip,
        compute_ref=not args.no_ref
    )
