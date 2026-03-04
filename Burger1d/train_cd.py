"""
Contrastive Divergence training for the energy-based Burgers equation solver.

The generator T_theta(u^n, xi) is trained to produce samples that match
Langevin-refined samples targeting the Gibbs distribution:
    P*(u^{n+1} | u^n) ∝ exp(-H(u^n, u^{n+1}))

The training objective (amortized CD) is:
    L_CD(theta) = E[||T_theta(u^n, xi) - stopgrad(u_K^{n+1})||^2]

where u_K is obtained by running K steps of Langevin dynamics from
the initial proposal T_theta(u^n, xi).

Langevin step:
    u' = u - step_size * grad H + noise_scale * noise
"""

import argparse
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from generator import Generator
from energy import compute_energy, compute_energy_gradient
from preconditioned_langevin import preconditioned_langevin_refine, FourierPreconditioner
from langevin import langevin_refine
from data import BurgersDataset
from reference_solver import solve_burgers_reference
from utils import compute_relative_l2_error, AverageMeter, count_parameters


def train_step(
    generator: nn.Module,
    u_curr: torch.Tensor,
    nu: float,
    dt: float,
    L: float,
    step_size: float,
    noise_scale: float,
    langevin_steps: int,
    optimizer: torch.optim.Optimizer,
    noise_dim: int,
    preconditioner: FourierPreconditioner = None
) -> dict:
    """
    Perform one training step of contrastive divergence.
    
    Args:
        generator: Generator network
        u_curr: Current solution batch, shape (batch, n)
        nu: Viscosity
        dt: Time step
        L: Domain length
        step_size: Langevin gradient step size
        noise_scale: Langevin noise magnitude
        langevin_steps: Number of Langevin steps
        optimizer: Optimizer for generator
        noise_dim: Dimension of noise input
        preconditioner: Optional FourierPreconditioner
        
    Returns:
        metrics: Dictionary with loss, energies, etc.
    """
    generator.train()
    batch_size = u_curr.shape[0]
    device = u_curr.device
    
    # Sample noise
    xi = torch.randn(batch_size, noise_dim, device=device)
    
    # Generate initial proposal for u^{n+1}
    u0 = generator(u_curr, xi)
    
    # Compute initial energy (for monitoring)
    with torch.no_grad():
        energy_init = compute_energy(u0, u_curr, nu, dt, L).mean()
    
    # Langevin refinement (no gradients through this)
    with torch.no_grad():
        if preconditioner is not None:
            u_refined = preconditioned_langevin_refine(
                u0.detach(), u_curr, nu, dt, L,
                step_size, noise_scale, langevin_steps, preconditioner
            )
        else:
            u_refined = langevin_refine(
                u0.detach(), u_curr, nu, dt, L,
                step_size, noise_scale, langevin_steps
            )
        
        energy_refined = compute_energy(u_refined, u_curr, nu, dt, L).mean()
    
    # CD loss: match generator output to refined samples
    loss = nn.functional.mse_loss(u0, u_refined.detach())
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'energy_init': energy_init.item(),
        'energy_refined': energy_refined.item()
    }


def evaluate(
    generator: nn.Module,
    dataloader: DataLoader,
    nu: float,
    dt: float,
    L: float,
    device: str,
    n_samples: int = 1
) -> dict:
    """
    Evaluate generator by computing mean energy AND L2 error vs reference.
    Expects dataloader to return (u_curr, u_ref_next).
    """
    h = L / generator.n_grid
    generator.eval()
    
    total_l2_error = 0.0
    total_energy_gap = 0.0
    total_count = 0
    
    with torch.no_grad():
        for u_curr, u_ref in dataloader:
            u_curr = u_curr.to(device)
            u_ref = u_ref.to(device)
            
            # Generate samples
            u_curr_expanded = u_curr.repeat_interleave(n_samples, dim=0)
            xi = torch.randn(u_curr_expanded.shape[0], generator.noise_dim, device=device)
            u_gen = generator(u_curr_expanded, xi)
            
            # Expand reference
            u_ref_expanded = u_ref.repeat_interleave(n_samples, dim=0)
            
            # L2 Error
            l2_err = compute_relative_l2_error(u_gen, u_ref_expanded, h)
            total_l2_error += l2_err.sum().item()
            
            # Energy gap
            energy_gen = compute_energy(u_gen, u_curr_expanded, nu, dt, L)
            energy_ref = compute_energy(u_ref_expanded, u_curr_expanded, nu, dt, L)
            total_energy_gap += (energy_gen - energy_ref).abs().sum().item()
            
            total_count += u_curr_expanded.shape[0]
    
    generator.train()
    
    return {
        'mean_l2_error': total_l2_error / total_count,
        'mean_energy_gap': total_energy_gap / total_count
    }


def main(args):
    # Setup
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Grid setup
    n_grid = args.n_grid
    L = args.L
    h = L / n_grid
    
    # Create datasets
    train_dataset = BurgersDataset(
        n_grid, args.n_train,
        method=args.ic_method,
        amplitude=args.ic_amplitude,
        length_scale=args.ic_length_scale,
        L=L,
        pregenerate=True
    )
    
    val_dataset = BurgersDataset(
        n_grid, args.n_val,
        method=args.ic_method,
        amplitude=args.ic_amplitude,
        length_scale=args.ic_length_scale,
        L=L,
        pregenerate=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Pre-compute validation reference solutions
    print(f"\nPre-computing validation reference solutions for {len(val_dataset)} samples...")
    t0 = time.time()
    u_val_curr = val_dataset.u_data
    
    u_ref_list = []
    chunk_size = 100
    
    for i in range(0, len(u_val_curr), chunk_size):
        u_chunk = u_val_curr[i:i + chunk_size]
        u_ref_chunk = solve_burgers_reference(
            u_chunk, args.nu, args.dt, L
        )
        u_ref_list.append(u_ref_chunk)
        if (i // chunk_size) % 5 == 0:
            print(f"  Processed {i}/{len(u_val_curr)} samples")
    
    u_ref_val = torch.cat(u_ref_list, dim=0)
    print(f"Validation pre-computation done in {time.time() - t0:.2f}s")
    
    # Create validation dataloader with (u_curr, u_ref) pairs
    val_dataset_full = TensorDataset(u_val_curr, u_ref_val)
    val_loader = DataLoader(val_dataset_full, batch_size=args.batch_size, shuffle=False)
    
    # Create generator
    generator = Generator(
        n_grid=n_grid,
        noise_dim=args.noise_dim,
        hidden_dims=[args.hidden_dim] * args.n_layers,
        activation=args.activation
    ).to(device)
    
    print(f"Generator parameters: {count_parameters(generator):,}")
    
    # Create preconditioner if enabled
    preconditioner = None
    if args.preconditioned:
        preconditioner = FourierPreconditioner(
            n_grid,
            kappa=args.precond_kappa,
            alpha=args.precond_alpha,
            L=L,
            device=device
        )
        print(f"Preconditioner: Fourier-Matérn (κ={args.precond_kappa}, α={args.precond_alpha})")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"cd_burgers_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining config:")
    print(f"  Grid: {n_grid}, L: {L:.4f}, h: {h:.4f}")
    print(f"  Physics: nu={args.nu}, dt={args.dt}")
    print(f"  Langevin: step_size={args.step_size}, noise_scale={args.noise_scale}, K={args.langevin_steps}")
    print(f"  IC method: {args.ic_method}, amplitude: {args.ic_amplitude}")
    print(f"  Training samples: {args.n_train}")
    print(f"  Output: {output_dir}")
    print()
    
    if args.dry_run:
        print("Dry run complete.")
        return
    
    # Metric tracking
    train_history = []
    val_history = []
    best_l2_error = float('inf')
    
    loss_meter = AverageMeter()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        loss_meter.reset()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, u_curr in enumerate(pbar):
            u_curr = u_curr.to(device)
            
            metrics = train_step(
                generator, u_curr,
                args.nu, args.dt, L,
                args.step_size, args.noise_scale,
                args.langevin_steps,
                optimizer, args.noise_dim,
                preconditioner=preconditioner
            )
            
            loss_meter.update(metrics['loss'], u_curr.shape[0])
            
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.6f}",
                'E_init': f"{metrics['energy_init']:.4f}",
                'E_ref': f"{metrics['energy_refined']:.4f}"
            })
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_history.append(loss_meter.avg)
        
        # Validation
        if (epoch + 1) % args.eval_interval == 0:
            print(f"\nRunning validation...")
            val_metrics = evaluate(
                generator, val_loader,
                args.nu, args.dt, L, device
            )
            
            val_history.append((epoch + 1, val_metrics['mean_l2_error']))
            
            print(f"\n[Epoch {epoch+1}] Val L2 Error: {val_metrics['mean_l2_error']:.6f}, "
                  f"Energy Gap: {val_metrics['mean_energy_gap']:.6f}, "
                  f"Time: {epoch_time:.1f}s\n")
            
            # Save best model
            if val_metrics['mean_l2_error'] < best_l2_error:
                best_l2_error = val_metrics['mean_l2_error']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': vars(args)
                }, os.path.join(output_dir, 'best_model.pt'))
                print(f"  Saved best model (L2 error: {best_l2_error:.6f})")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, os.path.join(output_dir, 'final_model.pt'))
    
    print(f"\nTraining complete. Best validation L2 error: {best_l2_error:.6f}")
    print(f"Models saved to: {output_dir}")
    
    # Plotting
    try:
        plt.figure(figsize=(10, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_history, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('CD Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Val Error
        if val_history:
            val_epochs, val_errors = zip(*val_history)
            plt.subplot(1, 2, 2)
            plt.plot(val_epochs, val_errors, 'r-', label='Val L2 Error')
            plt.xlabel('Epoch')
            plt.ylabel('Relative L2 Error')
            plt.title('Validation Error')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
        print(f"Saved learning curves to {os.path.join(output_dir, 'learning_curves.png')}")
        plt.close()
    except Exception as e:
        print(f"Failed to plot learning curves: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Burgers equation solver with Contrastive Divergence'
    )
    
    # Grid and physics
    parser.add_argument('--n_grid', type=int, default=64, help='Number of grid points')
    parser.add_argument('--L', type=float, default=1.0, help='Domain length')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    
    # Data
    parser.add_argument('--n_train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=500, help='Number of validation samples')
    parser.add_argument('--ic_method', type=str, default='mixed',
                        choices=['grf', 'sinusoidal', 'mixed'],
                        help='Initial condition sampling method')
    parser.add_argument('--ic_amplitude', type=float, default=1.0, help='IC amplitude')
    parser.add_argument('--ic_length_scale', type=float, default=0.3, help='IC GRF length scale')
    
    # Generator
    parser.add_argument('--noise_dim', type=int, default=16, help='Noise dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    
    # Langevin
    parser.add_argument('--step_size', type=float, default=1e-4,
                        help='Langevin gradient step size')
    parser.add_argument('--noise_scale', type=float, default=1e-3,
                        help='Langevin noise magnitude')
    parser.add_argument('--langevin_steps', type=int, default=50,
                        help='Number of Langevin steps')
    
    # Preconditioner
    parser.add_argument('--preconditioned', action='store_true', default=True,
                        help='Use preconditioned Langevin dynamics')
    parser.add_argument('--no_preconditioned', action='store_false', dest='preconditioned',
                        help='Disable preconditioned Langevin dynamics')
    parser.add_argument('--precond_kappa', type=float, default=1.0, help='Precond. kappa')
    parser.add_argument('--precond_alpha', type=float, default=2.0, help='Precond. alpha')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    # Misc
    parser.add_argument('--cpu', action='store_true', help='Disable CUDA')
    parser.add_argument('--dry_run', action='store_true', help='Quick test without training')
    
    args = parser.parse_args()
    main(args)
