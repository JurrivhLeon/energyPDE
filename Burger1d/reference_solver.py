"""
Pseudo-spectral reference solver for the 1D viscous Burgers equation.

Solves:
    u_t + u u_x = nu u_xx,    x in [0, L),  periodic BCs

over t in [0, T] given initial condition u(x, 0) = u0(x).

Spatial derivatives are computed spectrally (FFT). Time integration uses
ETDRK4 (Exponential Time Differencing Runge-Kutta, 4th order), which
treats the linear diffusion term exactly and the nonlinear advection term
explicitly. This is very accurate and stable for advection-diffusion PDEs.

Reference:
    Cox & Matthews, "Exponential Time Differencing for Stiff Systems",
    J. Comput. Phys. 176(2), 430-455 (2002).
    
    Kassam & Trefethen, "Fourth-Order Time Stepping for Stiff PDEs",
    SIAM J. Sci. Comput. 26(4), 1214-1233 (2005).
"""

import numpy as np
import torch


def solve_burgers_pseudospectral(
    u0: np.ndarray,
    nu: float,
    L: float = 1.0,
    T: float = 1.0,
    dt: float = 1e-4,
    n_save: int = 100,
    method: str = 'etdrk4',
    dealias: bool = True
) -> dict:
    """
    Solve the viscous Burgers equation using pseudo-spectral method.
    
    u_t + u u_x = nu u_xx,   x in [0, L), periodic BCs.
    
    Args:
        u0: Initial condition on periodic grid, shape (n,)
        nu: Viscosity coefficient
        L: Domain length
        T: Final time
        dt: Time step for integration
        n_save: Number of snapshots to save (evenly spaced in [0, T])
        method: Time integration method ('etdrk4' or 'rk4')
        dealias: Use 2/3 dealiasing rule for nonlinear term
        
    Returns:
        result: dict with keys:
            't': Time points, shape (n_save+1,)
            'u': Solution snapshots, shape (n_save+1, n)
            'x': Grid points, shape (n,)
    """
    n = len(u0)
    h = L / n
    
    # Grid
    x = np.linspace(0, L, n, endpoint=False)
    
    # Wavenumbers
    k = np.fft.fftfreq(n, d=L / n) * (2 * np.pi)
    
    # Dealiasing mask (2/3 rule): zero out the top 1/3 of modes
    if dealias:
        k_max = n // 3
        dealias_mask = np.ones(n)
        dealias_mask[k_max + 1: n - k_max] = 0.0
    else:
        dealias_mask = np.ones(n)
    
    if method == 'etdrk4':
        return _solve_etdrk4(u0, k, nu, dt, T, n_save, x, dealias_mask)
    elif method == 'rk4':
        return _solve_rk4(u0, k, nu, dt, T, n_save, x, dealias_mask)
    else:
        raise ValueError(f"Unknown method: {method}")


def _nonlinear_term_hat(u_hat, k, dealias_mask):
    """
    Compute FFT of the nonlinear term N(u) = -u * u_x in Fourier space.
    
    Uses dealiasing: both u and u_x are computed in physical space
    with the dealias mask applied to their Fourier coefficients.
    """
    # Dealiased u in physical space
    u = np.real(np.fft.ifft(dealias_mask * u_hat))
    
    # Dealiased u_x in physical space
    u_x_hat = 1j * k * u_hat
    u_x = np.real(np.fft.ifft(dealias_mask * u_x_hat))
    
    # Nonlinear term: -u * u_x (in physical space), then to Fourier
    N_hat = np.fft.fft(-u * u_x)
    
    return N_hat


def _solve_etdrk4(u0, k, nu, dt, T, n_save, x, dealias_mask):
    """
    ETDRK4 time integration.
    
    The PDE in Fourier space is:
        d(u_hat)/dt = L_hat * u_hat + N_hat(u_hat)
    
    where L_hat = -nu * k^2 (linear diffusion) and 
          N_hat = FFT(-u * u_x) (nonlinear advection).
    """
    n = len(u0)
    n_steps = int(np.round(T / dt))
    save_interval = max(1, n_steps // n_save)
    
    # Linear operator in Fourier space: L = -nu * k^2
    L_hat = -nu * k ** 2
    
    # ETDRK4 coefficients (Kassam & Trefethen)
    # To compute these stably, we use contour integrals in the complex plane
    E = np.exp(L_hat * dt)
    E2 = np.exp(L_hat * dt / 2)
    
    # Number of points on contour for computing phi functions
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # roots of unity
    
    # Compute ETDRK4 coefficients using contour integrals for numerical stability
    # LR shape: (n, M) - L_hat values shifted along contour
    LR = dt * L_hat[:, None] + r[None, :]  # (n, M)
    
    # phi_1(z) = (e^z - 1) / z
    # phi_2(z) = (e^z - 1 - z) / z^2
    # phi_3(z) = (e^z - 1 - z - z^2/2) / z^3
    
    # Q = dt * phi_1(L dt / 2)
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    
    # f1 = dt * phi_1(L dt) = dt * (e^{L dt} - 1) / (L dt)
    # But we need specific combinations for ETDRK4:
    # a = (-4 - L*dt + e^{L*dt} * (4 - 3*L*dt + (L*dt)^2)) / (L*dt)^3
    # b = (2 + L*dt + e^{L*dt} * (-2 + L*dt)) / (L*dt)^3
    # c = (-4 - 3*L*dt - (L*dt)^2 + e^{L*dt} * (4 - L*dt)) / (L*dt)^3
    
    f1 = dt * np.real(np.mean(
        (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1
    ))
    f2 = dt * np.real(np.mean(
        (2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1
    ))
    f3 = dt * np.real(np.mean(
        (-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1
    ))
    
    # Time integration
    u_hat = np.fft.fft(u0.copy())
    
    # Storage
    t_save = [0.0]
    u_save = [u0.copy()]
    
    t = 0.0
    for step in range(n_steps):
        # ETDRK4 stages
        Nu_hat = _nonlinear_term_hat(u_hat, k, dealias_mask)
        
        a_hat = E2 * u_hat + Q * Nu_hat
        Na_hat = _nonlinear_term_hat(a_hat, k, dealias_mask)
        
        b_hat = E2 * u_hat + Q * Na_hat
        Nb_hat = _nonlinear_term_hat(b_hat, k, dealias_mask)
        
        c_hat = E2 * a_hat + Q * (2 * Nb_hat - Nu_hat)
        Nc_hat = _nonlinear_term_hat(c_hat, k, dealias_mask)
        
        # Update
        u_hat = E * u_hat + Nu_hat * f1 + 2 * (Na_hat + Nb_hat) * f2 + Nc_hat * f3
        
        t += dt
        
        # Save snapshots
        if (step + 1) % save_interval == 0 or step == n_steps - 1:
            u_phys = np.real(np.fft.ifft(u_hat))
            t_save.append(t)
            u_save.append(u_phys.copy())
    
    return {
        't': np.array(t_save),
        'u': np.array(u_save),
        'x': x
    }


def _solve_rk4(u0, k, nu, dt, T, n_save, x, dealias_mask):
    """
    Classical RK4 time integration (explicit, simpler but may need smaller dt).
    """
    n = len(u0)
    n_steps = int(np.round(T / dt))
    save_interval = max(1, n_steps // n_save)
    
    L_hat = -nu * k ** 2
    
    def rhs_hat(u_hat_):
        """Full RHS in Fourier space: L u_hat + N_hat(u_hat)."""
        N_hat = _nonlinear_term_hat(u_hat_, k, dealias_mask)
        return L_hat * u_hat_ + N_hat
    
    u_hat = np.fft.fft(u0.copy())
    
    t_save = [0.0]
    u_save = [u0.copy()]
    
    t = 0.0
    for step in range(n_steps):
        k1 = dt * rhs_hat(u_hat)
        k2 = dt * rhs_hat(u_hat + k1 / 2)
        k3 = dt * rhs_hat(u_hat + k2 / 2)
        k4 = dt * rhs_hat(u_hat + k3)
        
        u_hat = u_hat + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        
        if (step + 1) % save_interval == 0 or step == n_steps - 1:
            u_phys = np.real(np.fft.ifft(u_hat))
            t_save.append(t)
            u_save.append(u_phys.copy())
    
    return {
        't': np.array(t_save),
        'u': np.array(u_save),
        'x': x
    }


def solve_burgers_reference(
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float = 1.0,
    solver_dt: float = 1e-4,
    method: str = 'etdrk4'
) -> torch.Tensor:
    """
    Solve one time step of Burgers equation using pseudo-spectral method.
    
    This is a drop-in replacement for the backward Euler root-finding solver.
    Given u(t), returns u(t + dt) by integrating the PDE with a fine dt_solver.
    
    Args:
        u_curr: Current solution, shape (batch, n) or (n,)
        nu: Viscosity
        dt: Time step to advance (target time = dt)
        L: Domain length
        solver_dt: Internal time step for the spectral solver
        method: 'etdrk4' or 'rk4'
        
    Returns:
        u_next: Solution at time dt, same shape as u_curr
    """
    squeeze_output = False
    if u_curr.dim() == 1:
        u_curr = u_curr.unsqueeze(0)
        squeeze_output = True
    
    batch_size, n = u_curr.shape
    results = []
    
    for i in range(batch_size):
        u0_np = u_curr[i].cpu().numpy()
        
        sol = solve_burgers_pseudospectral(
            u0_np, nu, L=L, T=dt, dt=solver_dt,
            n_save=1, method=method
        )
        
        # Take the last snapshot (t = dt)
        u_next_np = sol['u'][-1]
        results.append(torch.from_numpy(u_next_np).float())
    
    u_next = torch.stack(results, dim=0).to(u_curr.device)
    
    if squeeze_output:
        u_next = u_next.squeeze(0)
    
    return u_next


def solve_burgers_trajectory(
    u0: torch.Tensor,
    nu: float,
    L: float = 1.0,
    T: float = 1.0,
    dt: float = 1e-4,
    n_save: int = 100,
    method: str = 'etdrk4'
) -> dict:
    """
    Solve the Burgers equation over [0, T] and return the full trajectory.
    
    Convenience wrapper that handles torch <-> numpy conversion.
    
    Args:
        u0: Initial condition, shape (n,) or (batch, n)
        nu: Viscosity
        L: Domain length
        T: Final time
        dt: Solver time step
        n_save: Number of snapshots
        method: 'etdrk4' or 'rk4'
        
    Returns:
        result: dict with keys:
            't': Time points, shape (n_save+1,)
            'u': Solution snapshots, shape (n_save+1, n) or (n_save+1, batch, n)
            'x': Grid points, shape (n,)
    """
    if isinstance(u0, torch.Tensor):
        u0_np = u0.cpu().numpy()
    else:
        u0_np = np.asarray(u0)
    
    if u0_np.ndim == 1:
        return solve_burgers_pseudospectral(
            u0_np, nu, L, T, dt, n_save, method
        )
    else:
        # Batch: solve each sample independently
        batch_results = []
        for i in range(u0_np.shape[0]):
            res = solve_burgers_pseudospectral(
                u0_np[i], nu, L, T, dt, n_save, method
            )
            batch_results.append(res)
        
        return {
            't': batch_results[0]['t'],
            'u': np.stack([r['u'] for r in batch_results], axis=1),
            'x': batch_results[0]['x']
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    import time as time_mod
    
    parser = argparse.ArgumentParser(
        description='Test pseudo-spectral Burgers solver'
    )
    parser.add_argument('--n', type=int, default=128, help='Grid points')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity')
    parser.add_argument('--L', type=float, default=1.0, help='Domain length')
    parser.add_argument('--T', type=float, default=1.0, help='Final time')
    parser.add_argument('--dt', type=float, default=1e-4, help='Solver dt')
    parser.add_argument('--n_save', type=int, default=200, help='Snapshots')
    parser.add_argument('--method', type=str, default='etdrk4',
                        choices=['etdrk4', 'rk4'])
    args = parser.parse_args()
    
    n = args.n
    L = args.L
    x = np.linspace(0, L, n, endpoint=False)
    
    # Initial condition: sin(2*pi*x/L) -- single-mode, periodic
    u0 = np.sin(2 * np.pi * x / L)
    
    print(f"Solving Burgers equation with pseudo-spectral {args.method.upper()}")
    print(f"  n={n}, L={L}, nu={args.nu}, T={args.T}, dt={args.dt}")
    
    t0 = time_mod.time()
    result = solve_burgers_pseudospectral(
        u0, args.nu, L=L, T=args.T, dt=args.dt,
        n_save=args.n_save, method=args.method
    )
    elapsed = time_mod.time() - t0
    print(f"  Solved in {elapsed:.2f}s")
    
    t_arr = result['t']
    u_arr = result['u']
    print(f"  t shape: {t_arr.shape}, u shape: {u_arr.shape}")
    print(f"  t_min={t_arr[0]:.4f}, t_max={t_arr[-1]:.4f}")
    
    # Check conservation / dissipation
    for i, ti in enumerate(t_arr):
        if abs(ti) < 1e-12 or abs(ti - 0.01) < 1e-4 or abs(ti - 0.1) < 1e-3 or abs(ti - args.T) < 1e-4:
            L2 = np.sqrt(np.sum(u_arr[i] ** 2) * L / n)
            print(f"  t={ti:.4f}: ||u||_L2 = {L2:.6f}, max|u| = {np.max(np.abs(u_arr[i])):.6f}")
    
    # Plot space-time diagram
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Space-time contour
    ax = axes[0]
    T_mesh, X_mesh = np.meshgrid(t_arr, x)
    c = ax.pcolormesh(X_mesh, T_mesh, u_arr.T, shading='auto', cmap='RdBu_r')
    plt.colorbar(c, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'u(x,t) — ν={args.nu}')
    
    # Selected time snapshots
    ax = axes[1]
    n_curves = min(10, len(t_arr))
    indices = np.linspace(0, len(t_arr) - 1, n_curves, dtype=int)
    for idx in indices:
        ax.plot(x, u_arr[idx], label=f't={t_arr[idx]:.3f}', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Solution snapshots')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Energy dissipation
    ax = axes[2]
    energy = np.array([np.sum(u_arr[i] ** 2) * L / n for i in range(len(t_arr))])
    ax.plot(t_arr, energy)
    ax.set_xlabel('t')
    ax.set_ylabel('||u||²_L2')
    ax.set_title('Energy (L2 norm²) dissipation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = 'reference_solver_test.png'
    plt.savefig(fname, dpi=150)
    print(f"\nSaved plot to {fname}")
