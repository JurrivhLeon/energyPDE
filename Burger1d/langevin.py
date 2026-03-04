"""
Langevin dynamics for sample refinement in contrastive divergence training
for the 1D viscous Burgers equation.

The update rule uses two tunable parameters:
    u_{k+1} = u_k - step_size * grad_u H(u_k; u_curr) + noise_scale * epsilon_k

where epsilon_k ~ N(0, I) and H is the residual energy.
"""

import torch
import numpy as np
from energy import compute_energy_gradient


def langevin_step(
    u_next: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float,
    step_size: float,
    noise_scale: float = 0.0
) -> torch.Tensor:
    """
    Perform one step of Langevin dynamics.
    
    u' = u - step_size * grad H(u; u_curr) + noise_scale * noise
    
    Args:
        u_next: Current proposal for next-step solution, shape (batch, n) or (n,)
        u_curr: Current solution (fixed), shape (batch, n) or (n,)
        nu: Viscosity
        dt: Time step
        L: Domain length
        step_size: Gradient step size
        noise_scale: Noise magnitude (0 = deterministic gradient descent)
        
    Returns:
        u_updated: Updated proposal, same shape as u_next
    """
    grad = compute_energy_gradient(u_next, u_curr, nu, dt, L)
    noise = torch.randn_like(u_next)
    
    u_updated = u_next - step_size * grad + noise_scale * noise
    
    return u_updated


def langevin_refine(
    u0: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float,
    step_size: float,
    noise_scale: float = 0.0,
    K_steps: int = 50,
    return_trajectory: bool = False
) -> torch.Tensor:
    """
    Refine initial sample using K_steps of Langevin dynamics.
    
    Args:
        u0: Initial proposal, shape (batch, n) or (n,)
        u_curr: Current solution, shape (batch, n) or (n,)
        nu: Viscosity
        dt: Time step
        L: Domain length
        step_size: Gradient step size
        noise_scale: Noise magnitude
        K_steps: Number of Langevin steps
        return_trajectory: If True, return all intermediate samples
        
    Returns:
        u_K: Refined solution after K_steps
        trajectory: (optional) List of all samples if return_trajectory=True
    """
    u = u0.clone()
    
    if return_trajectory:
        trajectory = [u.clone()]
    
    for _ in range(K_steps):
        u = langevin_step(u, u_curr, nu, dt, L, step_size, noise_scale)
        if return_trajectory:
            trajectory.append(u.clone())
    
    if return_trajectory:
        return u, trajectory
    return u


def langevin_refine_with_energy(
    u0: torch.Tensor,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float,
    step_size: float,
    noise_scale: float = 0.0,
    K_steps: int = 50
) -> tuple:
    """
    Refine sample and track energy along the trajectory.
    
    Returns:
        u_K: Refined solution
        energies: List of energy values at each step
    """
    from energy import compute_energy
    
    u = u0.clone()
    energies = [compute_energy(u, u_curr, nu, dt, L).detach()]
    
    for _ in range(K_steps):
        u = langevin_step(u, u_curr, nu, dt, L, step_size, noise_scale)
        energies.append(compute_energy(u, u_curr, nu, dt, L).detach())
    
    return u, energies


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from energy import compute_energy
    
    torch.manual_seed(42)
    
    n = 64
    L = 1.0
    nu = 0.01
    dt = 0.01
    
    x = torch.linspace(0, L, n + 1)[:-1]
    
    # Smooth current solution (periodic on [0, L])
    u_curr = torch.sin(2 * np.pi * x / L)
    
    # Start from random proposal for next step
    u0 = torch.randn(n) * 0.5
    
    # Gradient descent (deterministic) to verify energy decrease
    step_size = 1e-4
    K_steps = 500
    
    print("=== Deterministic gradient descent ===")
    u = u0.clone()
    energies_gd = [compute_energy(u, u_curr, nu, dt, L).item()]
    
    for _ in range(K_steps):
        grad = compute_energy_gradient(u, u_curr, nu, dt, L)
        u = u - step_size * grad
        energies_gd.append(compute_energy(u, u_curr, nu, dt, L).item())
    
    print(f"  Initial energy: {energies_gd[0]:.6f}")
    print(f"  Final energy: {energies_gd[-1]:.6f}")
    print(f"  Energy decreased: {energies_gd[-1] < energies_gd[0]}")
    
    # Langevin refinement (with noise)
    print("\n=== Langevin refinement ===")
    u_refined, energies = langevin_refine_with_energy(
        u0, u_curr, nu, dt, L,
        step_size=1e-4, noise_scale=1e-3, K_steps=500
    )
    
    print(f"  Initial energy: {energies[0].item():.6f}")
    print(f"  Final energy: {energies[-1].item():.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Energy trajectory
    axes[0].plot(energies_gd, label='Gradient descent')
    axes[0].plot([e.item() for e in energies], label='Langevin', alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Energy H')
    axes[0].set_title('Energy during refinement')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Initial vs refined solution
    axes[1].plot(x, u0, 'r--', label='Initial (random)', alpha=0.7)
    axes[1].plot(x, u_refined, 'b-', label='Langevin refined')
    axes[1].plot(x, u_curr, 'k--', label='u_curr', alpha=0.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x)')
    axes[1].set_title('Solution refinement')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gradient descent solution
    axes[2].plot(x, u, 'g-', label='GD refined')
    axes[2].plot(x, u_curr, 'k--', label='u_curr', alpha=0.5)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('u(x)')
    axes[2].set_title('Gradient descent solution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('langevin_test_burgers.png', dpi=150)
    print("Saved langevin_test_burgers.png")
