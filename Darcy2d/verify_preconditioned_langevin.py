"""
Verification script for preconditioned Langevin dynamics on 2D Darcy flow.

Tests both:
(i) Deterministic preconditioned Langevin (preconditioned gradient descent)
(ii) Stochastic preconditioned Langevin

Uses LBFGS to find the true energy minimizer as reference.

Simplified parameterization:
    - step_size: controls gradient descent magnitude
    - noise_scale: controls stochastic noise magnitude (0 = deterministic)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from energy import compute_energy, compute_energy_gradient, pad_with_bc
from data import sample_coefficient_a, sample_forcing_f
from utils import compute_relative_l2_error_2d
from preconditioned_langevin import (
    MaternPreconditioner2d, 
    preconditioned_langevin_step,
)


def find_energy_minimum(a, f, h, max_iter=200):
    """Find true energy minimizer using LBFGS."""
    N = f.shape[-1]
    device = f.device
    u = torch.zeros(1, N, N, device=device, requires_grad=True)
    
    optimizer = torch.optim.LBFGS([u], max_iter=20, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        loss = compute_energy(u, a, f, h, beta=1.0)
        loss.backward()
        return loss
    
    for _ in range(max_iter // 20):
        optimizer.step(closure)
    
    return u.detach()


def run_preconditioned_langevin(
    u0, a, f, h, step_size, noise_scale, K, preconditioner
):
    """Run preconditioned Langevin and track energy."""
    u = u0.clone()
    energies = [compute_energy(u, a, f, h, beta=1.0).item()]
    
    for _ in range(K):
        u = preconditioned_langevin_step(
            u, a, f, h, step_size, noise_scale, preconditioner
        )
        energies.append(compute_energy(u, a, f, h, beta=1.0).item())
    
    return u, energies


def main():
    torch.manual_seed(42)
    
    # Configuration
    N = 64
    h = 1.0 / (N + 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Grid: {N}×{N}")
    
    # Simplified parameters
    K = 1000  # Number of steps
    
    # Step sizes for each method
    step_size_det = 10.0    # Deterministic gradient step
    step_size_sto = 10.0    # Stochastic gradient step
    noise_scale_sto = 1e-4 # Noise magnitude for stochastic
    
    # Preconditioner parameters
    kappa = 1.0
    alpha = 1.0
    
    # Sample problem
    a = sample_coefficient_a(N, 1, method='constant', device=device)
    f = sample_forcing_f(N, 1, method='mixed', amplitude=15.0, device=device)
    
    print(f"\nPermeability range: [{a.min().item():.3f}, {a.max().item():.3f}]")
    print(f"Forcing range: [{f.min().item():.3f}, {f.max().item():.3f}]")
    
    # Find true energy minimizer with LBFGS
    print("\nFinding energy minimizer with LBFGS...")
    u_opt = find_energy_minimum(a, f, h, max_iter=200)
    energy_opt = compute_energy(u_opt, a, f, h, beta=1.0).item()
    grad_opt = compute_energy_gradient(u_opt, a, f, h, beta=1.0)
    print(f"Optimal energy: {energy_opt:.6f}")
    print(f"Gradient at optimum (should be ~0): {grad_opt.norm().item():.6e}")
    print(f"Solution range: [{u_opt.min().item():.4f}, {u_opt.max().item():.4f}]")
    
    # Create preconditioner
    precond = MaternPreconditioner2d(
        N, kappa=kappa, alpha=alpha,
        mode='inverse_laplacian',
        device=device
    )
    print(f"\nPreconditioner: mode=inverse_laplacian, κ={kappa}, α={alpha}")
    
    # Initial guess (zero)
    u0 = torch.zeros(1, N, N, device=device)
    
    print(f"\nRunning {K} steps")
    
    # (i) Deterministic preconditioned Langevin
    print(f"\n--- Deterministic (step_size={step_size_det}, noise_scale=0) ---")
    u_det, energies_det = run_preconditioned_langevin(
        u0.clone(), a, f, h, step_size_det, 0.0, K, precond
    )
    l2_err_det = compute_relative_l2_error_2d(u_det, u_opt, h).item()
    print(f"Final energy: {energies_det[-1]:.6f} (opt: {energy_opt:.6f})")
    print(f"L2 error vs optimum: {l2_err_det:.4f}")
    
    # (ii) Stochastic preconditioned Langevin
    print(f"\n--- Stochastic (step_size={step_size_sto}, noise_scale={noise_scale_sto}) ---")
    u_sto, energies_sto = run_preconditioned_langevin(
        u0.clone(), a, f, h, step_size_sto, noise_scale_sto, K, precond
    )
    l2_err_sto = compute_relative_l2_error_2d(u_sto, u_opt, h).item()
    print(f"Final energy: {energies_sto[-1]:.6f}")
    print(f"L2 error vs optimum: {l2_err_sto:.4f}")
    
    # ============ Plotting ============
    fig = plt.figure(figsize=(25, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # Row 1: Input and reference
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(a[0, 1:-1, 1:-1].cpu().numpy(), cmap='RdBu_r', origin='lower')
    ax1.set_title('(a) Permeability $a(x,y)$')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(f[0].cpu().numpy(), cmap='RdBu_r', origin='lower')
    ax2.set_title('(b) Forcing $f(x,y)$')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    u_opt_full = pad_with_bc(u_opt)  # Add zero boundaries
    im3 = ax3.imshow(u_opt_full[0].cpu().numpy(), cmap='RdBu_r', origin='lower')
    ax3.set_title('(c) LBFGS Optimum $u^*$\n(with Dirichlet BC)')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Energy curves
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(energies_det, 'b-', label='Deterministic', alpha=0.8)
    ax4.plot(energies_sto, 'r-', label='Stochastic', alpha=0.8)
    ax4.axhline(y=energy_opt, color='k', linestyle='--', label='Optimum', linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Energy $J(u)$')
    ax4.set_title('(d) Energy Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Row 2: Deterministic results
    ax5 = fig.add_subplot(gs[1, 0])
    u_det_full = pad_with_bc(u_det)
    im5 = ax5.imshow(u_det_full[0].cpu().numpy(), cmap='RdBu_r', origin='lower')
    ax5.set_title(f'(e) Deterministic Solution\n(L² err = {l2_err_det:.4f})')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    err_det = (u_det[0] - u_opt[0]).cpu().numpy()
    vmax = max(abs(err_det.min()), abs(err_det.max()), 1e-6)
    im6 = ax6.imshow(err_det, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    ax6.set_title('(f) Deterministic Error')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Stochastic results
    ax7 = fig.add_subplot(gs[1, 2])
    u_sto_full = pad_with_bc(u_sto)
    im7 = ax7.imshow(u_sto_full[0].cpu().numpy(), cmap='RdBu_r', origin='lower')
    ax7.set_title(f'(g) Stochastic Solution\n(L² err = {l2_err_sto:.4f})')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[1, 3])
    err_sto = (u_sto[0] - u_opt[0]).cpu().numpy()
    vmax = max(abs(err_sto.min()), abs(err_sto.max()), 1e-6)
    im8 = ax8.imshow(err_sto, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    ax8.set_title('(h) Stochastic Error')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Row 3: Cross sections and L2 error trajectory
    ax9 = fig.add_subplot(gs[2, 0:2])
    mid = N // 2
    x = np.linspace(0, 1, N)
    ax9.plot(x, u_opt[0, mid, :].cpu().numpy(), 'k-', linewidth=2, label='Optimum')
    ax9.plot(x, u_det[0, mid, :].cpu().numpy(), 'b--', label='Deterministic')
    ax9.plot(x, u_sto[0, mid, :].cpu().numpy(), 'r--', label='Stochastic')
    ax9.set_xlabel('x')
    ax9.set_ylabel('u(x, y=0.5)')
    ax9.set_title('(i) Cross-section at y = 0.5')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # L2 error vs step
    ax10 = fig.add_subplot(gs[2, 2:4])
    
    # Track L2 error
    u_track = u0.clone()
    l2_det = [compute_relative_l2_error_2d(u_track, u_opt, h).item()]
    for i in range(K):
        u_track = preconditioned_langevin_step(
            u_track, a, f, h, step_size_det, 0.0, precond
        )
        if (i + 1) % 20 == 0:
            l2_det.append(compute_relative_l2_error_2d(u_track, u_opt, h).item())
    
    torch.manual_seed(42)
    u_track = u0.clone()
    l2_sto = [compute_relative_l2_error_2d(u_track, u_opt, h).item()]
    for i in range(K):
        u_track = preconditioned_langevin_step(
            u_track, a, f, h, step_size_sto, noise_scale_sto, precond
        )
        if (i + 1) % 20 == 0:
            l2_sto.append(compute_relative_l2_error_2d(u_track, u_opt, h).item())
    
    steps = [0] + list(range(20, K + 1, 20))
    ax10.semilogy(steps, l2_det, 'b-', label='Deterministic')
    ax10.semilogy(steps, l2_sto, 'r-', label='Stochastic')
    ax10.set_xlabel('Step')
    ax10.set_ylabel('Relative L² Error')
    ax10.set_title('(j) L² Error vs Step')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    plt.suptitle(f'Preconditioned Langevin Verification ({N}×{N} grid, K={K})', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig('verification_preconditioned_langevin.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to verification_preconditioned_langevin.png")
    plt.close()


if __name__ == "__main__":
    main()
