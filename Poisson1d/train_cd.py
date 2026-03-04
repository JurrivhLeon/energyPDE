"""
Contrastive Divergence training for the energy-based PDE solver.

The generator T_theta(a, f, xi) is trained to produce samples that match
Langevin-refined samples targeting the Gibbs distribution:
    P*(u|a,f) ∝ exp(-J(u; a, f))

Training objective (Amortized CD):
    L_CD(theta) = E[||T_theta(a, f, xi) - stopgrad(u_K)||^2]

where u_K is obtained by running K steps of Langevin dynamics from the
initial proposal T_theta(a, f, xi).
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from energy import compute_energy, compute_energy_gradient
from generator import Generator, ConditionalGenerator
from langevin import langevin_refine
from preconditioned_langevin import preconditioned_langevin_refine, MaternPreconditioner
from data import PoissonDataset
from utils import solve_poisson_exact, compute_relative_l2_error, AverageMeter, count_parameters


def train_step(
    generator: nn.Module,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    langevin_steps: int,
    optimizer: torch.optim.Optimizer,
    noise_dim: int,
    preconditioner: MaternPreconditioner = None
) -> dict:
    """
    Perform one training step of contrastive divergence.
    
    Args:
        generator: Generator network
        a: Coefficient batch, shape (batch, n+2)
        f: Forcing batch, shape (batch, n)
        h: Grid spacing
        step_size: Langevin step size
        noise_scale: Langevin noise magnitude
        langevin_steps: Number of Langevin steps
        optimizer: Optimizer for generator
        noise_dim: Dimension of noise input
        preconditioner: Optional MaternPreconditioner
        
    Returns:
        metrics: Dictionary with loss, energies, etc.
    """
    batch_size = a.shape[0]
    device = a.device
    
    # Sample noise
    xi = torch.randn(batch_size, noise_dim, device=device)
    
    # Generate initial proposal
    u0 = generator(a, f, xi)
    
    # Compute initial energy (for monitoring)
    with torch.no_grad():
        energy_init = compute_energy(u0, a, f, h).mean()
    
    # Langevin refinement (no gradients through this)
    with torch.no_grad():
        if preconditioner is not None:
            u_refined = preconditioned_langevin_refine(
                u0.detach(), a, f, h, step_size, noise_scale, langevin_steps, preconditioner
            )
        else:
            u_refined = langevin_refine(u0.detach(), a, f, h, step_size, noise_scale, langevin_steps)
        energy_refined = compute_energy(u_refined, a, f, h).mean()
    
    # Compute CD loss: MSE between initial proposal and refined sample
    loss = nn.functional.mse_loss(u0, u_refined)
    
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
    h: float,
    device: str,
    n_samples: int = 5,
) -> dict:
    """
    Evaluate generator against exact solutions using multiple samples per input.
    """
    generator.eval()
    
    total_l2_error = 0.0
    total_energy_error = 0.0
    total_count = 0
    
    with torch.no_grad():
        for a, f in dataloader:
            a = a.to(device)
            f = f.to(device)
            batch_size = a.shape[0]
            
            # Repeat inputs for multiple samples
            # a: (batch, n+2) -> (batch * n_samples, n+2)
            # f: (batch, n) -> (batch * n_samples, n)
            a_expanded = a.repeat_interleave(n_samples, dim=0)
            f_expanded = f.repeat_interleave(n_samples, dim=0)
            
            # Generate solutions
            u_gen = generator(a_expanded, f_expanded)
            
            # Exact solution (computed once per batch item, then repeated)
            u_exact = solve_poisson_exact(a, f, h)
            u_exact_expanded = u_exact.repeat_interleave(n_samples, dim=0)
            
            # Compute independent errors for each sample
            l2_err = compute_relative_l2_error(u_gen, u_exact_expanded, h)
            total_l2_error += l2_err.sum().item()
            
            # Energy comparison
            energy_gen = compute_energy(u_gen, a_expanded, f_expanded, h)
            energy_exact = compute_energy(u_exact_expanded, a_expanded, f_expanded, h)
            total_energy_error += (energy_gen - energy_exact).abs().sum().item()
            
            total_count += batch_size * n_samples
    
    generator.train()
    
    return {
        'mean_l2_error': total_l2_error / total_count,
        'mean_energy_gap': total_energy_error / total_count
    }


def main(args):
    # Setup
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Grid
    n_interior = args.n_grid
    h = 1.0 / (n_interior + 1)
    
    # Create datasets
    train_dataset = PoissonDataset(
        n_interior=n_interior,
        n_samples=args.n_train,
        a_method=args.a_method,
        f_method=args.f_method,
        a_min=args.a_min,
        a_max=args.a_max,
        f_amplitude=args.f_amplitude,
        pregenerate=True
    )
    val_dataset = PoissonDataset(
        n_interior=n_interior,
        n_samples=args.n_val,
        a_method=args.a_method,
        f_method=args.f_method,
        a_min=args.a_min,
        a_max=args.a_max,
        f_amplitude=args.f_amplitude,
        pregenerate=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create generator
    if args.generator == 'simple':
        generator = Generator(
            n_grid=n_interior,
            noise_dim=args.noise_dim,
            hidden_dims=[args.hidden_dim] * args.n_layers
        )
    else:
        generator = ConditionalGenerator(
            n_grid=n_interior,
            noise_dim=args.noise_dim
        )
    generator = generator.to(device)
    print(f"Generator parameters: {count_parameters(generator):,}")
    
    # Create preconditioner if enabled
    preconditioner = None
    if args.preconditioned:
        preconditioner = MaternPreconditioner(
            n_grid=n_interior,
            kappa=args.precond_kappa,
            alpha=args.precond_alpha,
            normalize=True,
            mode=args.precond_mode,
            device=device
        )
        print(f"Using preconditioned Langevin: mode={args.precond_mode}, "
              f"κ={args.precond_kappa}, α={args.precond_alpha}")
    
    # Optimizer
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    loss_meter = AverageMeter()
    best_l2_error = float('inf')
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"cd_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining config:")
    print(f"  Grid points: {n_interior}, h = {h:.4f}")
    print(f"  Langevin: step_size={args.step_size}, noise_scale={args.noise_scale}, K={args.langevin_steps}")
    if args.preconditioned:
        print(f"  Preconditioner: {args.precond_mode} (κ={args.precond_kappa}, α={args.precond_alpha})")
    print(f"  Training samples: {args.n_train}")
    print(f"  Forcing Amplitude: {args.f_amplitude}")
    print(f"  Output: {output_dir}")
    print()
    
    if args.dry_run:
        print("Dry run complete.")
        return
    
    # Metric tracking
    train_history = []
    val_history = []
    
    from tqdm import tqdm

    for epoch in range(args.epochs):
        epoch_start = time.time()
        loss_meter.reset()
        
        # Progress bar for the epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (a, f) in enumerate(pbar):
            a = a.to(device)
            f = f.to(device)
            
            metrics = train_step(
                generator=generator,
                a=a,
                f=f,
                h=h,
                step_size=args.step_size,
                noise_scale=args.noise_scale,
                langevin_steps=args.langevin_steps,
                optimizer=optimizer,
                noise_dim=args.noise_dim,
                preconditioner=preconditioner
            )
            
            loss_meter.update(metrics['loss'], a.shape[0])
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.6f}",
                'E_init': f"{metrics['energy_init']:.2f}",
                'E_ref': f"{metrics['energy_refined']:.2f}"
            })
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Record training loss
        train_history.append(loss_meter.avg)
        
        # Validation
        if (epoch + 1) % args.eval_interval == 0:
            print(f"\nRunning validation...")
            val_metrics = evaluate(generator, val_loader, h, device)
            
            # Record validation metric
            val_history.append((epoch + 1, val_metrics['mean_l2_error']))
            
            print(f"\n[Epoch {epoch+1}] Val L2 Error: {val_metrics['mean_l2_error']:.6f}, "
                  f"Energy Gap: {val_metrics['mean_energy_gap']:.6f}, "
                  f"Time: {epoch_time:.1f}s\n")
            
            # Save best
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
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
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
    
    print(f"\nTraining complete. Best L2 error: {best_l2_error:.6f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train energy-based PDE solver with Contrastive Divergence")
    
    # Grid
    parser.add_argument('--n_grid', type=int, default=64, help='Number of interior grid points')
    
    # Data
    parser.add_argument('--n_train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--a_method', type=str, default='constant', choices=['grf', 'constant', 'piecewise'])
    parser.add_argument('--f_method', type=str, default='grf', choices=['grf', 'sinusoidal', 'polynomial'])
    parser.add_argument('--f_amplitude', type=float, default=20.0, help='Amplitude of forcing term')
    parser.add_argument('--a_min', type=float, default=0.1, help='Minimum coefficient value')
    parser.add_argument('--a_max', type=float, default=2.0, help='Maximum coefficient value')
    
    # Model
    parser.add_argument('--generator', type=str, default='simple', choices=['simple', 'conditional'])
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--noise_dim', type=int, default=16, help='Noise dimension')
    
    # Energy / Langevin
    parser.add_argument('--step_size', type=float, default=1e-4, help='Langevin step size')
    parser.add_argument('--noise_scale', type=float, default=1e-3, help='Langevin noise magnitude')
    parser.add_argument('--langevin_steps', type=int, default=50, help='Number of Langevin steps')
    
    # Preconditioned Langevin
    parser.add_argument('--preconditioned', type=bool, default=True, help='Use preconditioned Langevin dynamics')
    parser.add_argument('--precond_mode', type=str, default='matern', 
                        choices=['matern', 'inverse_laplacian'], help='Preconditioner mode')
    parser.add_argument('--precond_kappa', type=float, default=1.0, help='Preconditioner κ parameter')
    parser.add_argument('--precond_alpha', type=float, default=2.0, help='Preconditioner α parameter')
    
    # Training
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=20, help='Log every N batches')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    # Misc
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--dry_run', action='store_true', help='Quick test without training')
    
    args = parser.parse_args()
    main(args)
