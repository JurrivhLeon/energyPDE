"""
Langevin dynamics for sample refinement in contrastive divergence training.

The update rule uses two tunable parameters:
    u_{k+1} = u_k - step_size * grad_u J(u_k) + noise_scale * epsilon_k

where epsilon_k ~ N(0, I).
"""

import torch
from energy import compute_energy_gradient


def langevin_step(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float = 0.0
) -> torch.Tensor:
    """
    Perform one step of Langevin dynamics.
    
    u' = u - step_size * grad J(u) + noise_scale * noise
    
    Args:
        u: Current solution, shape (batch, n) or (n,)
        a: Coefficient function, shape (batch, n+2) or (n+2,)
        f: Forcing term, shape (batch, n) or (n,)
        h: Grid spacing
        step_size: Gradient step size
        noise_scale: Noise magnitude (0 = deterministic gradient descent)
        
    Returns:
        u_next: Updated solution, same shape as u
    """
    grad = compute_energy_gradient(u, a, f, h)
    noise = torch.randn_like(u)
    
    u_next = u - step_size * grad + noise_scale * noise
    
    return u_next


def langevin_refine(
    u0: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float = 0.0,
    K: int = 50,
    return_trajectory: bool = False
) -> torch.Tensor:
    """
    Refine initial sample using K steps of Langevin dynamics.
    
    Args:
        u0: Initial solution, shape (batch, n) or (n,)
        a: Coefficient function, shape (batch, n+2) or (n+2,)
        f: Forcing term, shape (batch, n) or (n,)
        h: Grid spacing
        step_size: Gradient step size
        noise_scale: Noise magnitude
        K: Number of Langevin steps
        return_trajectory: If True, return all intermediate samples
        
    Returns:
        u_K: Refined solution after K steps
        trajectory: (optional) List of all samples if return_trajectory=True
    """
    u = u0.clone()
    
    if return_trajectory:
        trajectory = [u.clone()]
    
    for _ in range(K):
        u = langevin_step(u, a, f, h, step_size, noise_scale)
        if return_trajectory:
            trajectory.append(u.clone())
    
    if return_trajectory:
        return u, trajectory
    return u


def langevin_refine_with_energy(
    u0: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float = 0.0,
    K: int = 50
) -> tuple:
    """
    Refine sample and track energy along the trajectory.
    
    Returns:
        u_K: Refined solution
        energies: List of energy values at each step
    """
    from energy import compute_energy
    
    u = u0.clone()
    energies = [compute_energy(u, a, f, h).detach()]
    
    for _ in range(K):
        u = langevin_step(u, a, f, h, step_size, noise_scale)
        energies.append(compute_energy(u, a, f, h).detach())
    
    return u, energies


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from energy import compute_energy
    
    torch.manual_seed(42)
    
    n = 20
    h = 1.0 / (n + 1)
    
    # Standard Poisson with sinusoidal forcing
    a = torch.ones(n + 2)
    f = torch.sin(torch.linspace(0, 3.14159, n))
    
    # Start from random
    u0 = torch.randn(n)
    
    # Refine with Langevin
    step_size = 1e-4
    noise_scale = 1e-3
    K = 500
    
    u_refined, energies = langevin_refine_with_energy(u0, a, f, h, step_size, noise_scale, K)
    
    print(f"Initial energy: {energies[0].item():.4f}")
    print(f"Final energy: {energies[-1].item():.4f}")
    
    # Plot energy trajectory
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(energies)
    plt.xlabel("Langevin step")
    plt.ylabel("Energy J(u)")
    plt.title("Energy during Langevin refinement")
    
    # Plot solution
    x = torch.linspace(0, 1, n + 2)
    u_full_init = torch.cat([torch.zeros(1), u0, torch.zeros(1)])
    u_full_refined = torch.cat([torch.zeros(1), u_refined, torch.zeros(1)])
    
    plt.subplot(1, 2, 2)
    plt.plot(x, u_full_init, 'r--', label='Initial (random)')
    plt.plot(x, u_full_refined, 'b-', label='Refined')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution refinement")
    
    plt.tight_layout()
    plt.savefig("langevin_test.png", dpi=150)
    print("Saved plot to langevin_test.png")
