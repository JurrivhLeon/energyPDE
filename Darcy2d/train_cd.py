"""
Contrastive Divergence training for 2D Darcy flow energy-based solver.

The generator T_theta(a, f, xi) is trained to produce samples that match
Langevin-refined samples targeting the Gibbs distribution:
    P*(u|a,f) ∝ exp(-beta * J(u; a, f))

Training objective (Amortized CD):
    L_CD(theta) = E[||T_theta(a, f, xi) - stopgrad(u_K)||^2]

where u_K is obtained by running K steps of preconditioned Langevin dynamics.
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
from generator import Generator2d, MLPGenerator2d, FNOGenerator2d
from preconditioned_langevin import preconditioned_langevin_refine, MaternPreconditioner2d
from data import DarcyDataset
from utils import solve_darcy_exact, compute_relative_l2_error_2d, AverageMeter, count_parameters


def train_step(
    generator: nn.Module,
    a: torch.Tensor,
    f: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    langevin_steps: int,
    optimizer: torch.optim.Optimizer,
    preconditioner: MaternPreconditioner2d,
    noise_channels: int = 8
) -> dict:
    """
    Perform one training step of contrastive divergence.
    
    Args:
        generator: Generator network
        a: Permeability batch, shape (batch, N+2, N+2)
        f: Forcing batch, shape (batch, N, N)
        h: Grid spacing
        step_size: Langevin gradient step size
        noise_scale: Langevin noise magnitude (0 = deterministic)
        langevin_steps: Number of Langevin steps
        optimizer: Optimizer for generator
        preconditioner: MaternPreconditioner2d for Langevin dynamics
        noise_channels: Number of noise channels for generator
        
    Returns:
        metrics: Dictionary with loss, energies, etc.
    """
    batch_size = a.shape[0]
    device = a.device
    N = f.shape[-1]
    
    # Sample noise for generator
    xi = torch.randn(batch_size, noise_channels, N, N, device=device)
    
    # Generate initial proposal
    u0 = generator(a, f, xi)
    
    # Compute initial energy (for monitoring)
    with torch.no_grad():
        energy_init = compute_energy(u0, a, f, h, beta=1.0).mean()
    
    # Langevin refinement (no gradients through this)
    with torch.no_grad():
        u_refined = preconditioned_langevin_refine(
            u0.detach(), a, f, h, step_size, noise_scale, langevin_steps, preconditioner
        )
        energy_refined = compute_energy(u_refined, a, f, h, beta=1.0).mean()
    
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


def precompute_validation_data(dataloader, h, device):
    """Pre-compute exact solutions for validation set."""
    data = []
    from tqdm import tqdm
    print("Pre-computing validation solutions...")
    # Using tqdm for progress tracking
    for a, f in tqdm(dataloader, desc="Pre-computing"):
        a = a.to(device)
        f = f.to(device)
        with torch.no_grad():
            u_exact = solve_darcy_exact(a, f, h)
        data.append((a, f, u_exact))
    return data


def evaluate(
    generator: nn.Module,
    validation_data: list,
    h: float,
    device: str,
    n_samples: int = 1,
) -> dict:
    """
    Evaluate generator against exact solutions using pre-computed data.
    """
    generator.eval()
    
    total_l2_error = 0.0
    total_energy_error = 0.0
    total_count = 0
    
    with torch.no_grad():
        for a, f, u_exact in validation_data:
            # a, f, u_exact are already on device
            batch_size = a.shape[0]
            
            # Expand for multiple samples
            a_expanded = a.repeat_interleave(n_samples, dim=0)
            f_expanded = f.repeat_interleave(n_samples, dim=0)
            u_exact_expanded = u_exact.repeat_interleave(n_samples, dim=0)
            
            # Generate solutions
            u_gen = generator(a_expanded, f_expanded)
            
            # Compute errors
            l2_err = compute_relative_l2_error_2d(u_gen, u_exact_expanded, h)
            total_l2_error += l2_err.sum().item()
            
            # Energy comparison
            energy_gen = compute_energy(u_gen, a_expanded, f_expanded, h, beta=1.0)
            energy_exact = compute_energy(u_exact_expanded, a_expanded, f_expanded, h, beta=1.0)
            total_energy_error += (energy_gen - energy_exact).abs().sum().item()
            
            total_count += batch_size * n_samples
    
    generator.train()
    
    return {
        'mean_l2_error': total_l2_error / total_count,
        'mean_energy_gap': total_energy_error / total_count
    }


def main(args):
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Grid
    N = args.n_grid
    h = 1.0 / (N + 1)
    
    # Create datasets
    train_dataset = DarcyDataset(
        N=N,
        n_samples=args.n_train,
        a_method=args.a_method,
        f_method=args.f_method,
        a_min=args.a_min,
        a_max=args.a_max,
        f_amplitude=args.f_amplitude,
        pregenerate=True
    )
    val_dataset = DarcyDataset(
        N=N,
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
    
    # Pre-compute validation solutions
    print(f"\nPre-computing {len(val_dataset)} validation solutions...")
    val_data_precomputed = precompute_validation_data(val_loader, h, device)
    
    # Create generator
    if args.generator == 'cnn':
        generator = Generator2d(
            N=N,
            noise_channels=args.noise_channels,
            base_channels=args.base_channels,
            depth=args.depth
        )
    elif args.generator == 'fno':
        generator = FNOGenerator2d(
            N=N,
            noise_channels=args.noise_channels,
            width=args.fno_width,
            modes=args.fno_modes,
            n_layers=args.fno_layers
        )
    else:
        generator = MLPGenerator2d(
            N=N,
            noise_dim=args.noise_channels * N * N,
            hidden_dims=[args.hidden_dim] * args.n_layers
        )
    generator = generator.to(device)
    print(f"Generator parameters: {count_parameters(generator):,}")
    
    # Create preconditioner
    preconditioner = MaternPreconditioner2d(
        N=N,
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
    output_dir = os.path.join(args.output_dir, f"darcy2d_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining config:")
    print(f"  Grid: {N}×{N}, h = {h:.4f}")
    print(f"  Langevin: step_size={args.step_size}, noise_scale={args.noise_scale}, K={args.langevin_steps}")
    print(f"  Preconditioner: {args.precond_mode} (κ={args.precond_kappa}, α={args.precond_alpha})")
    print(f"  Training samples: {args.n_train}")
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
                preconditioner=preconditioner,
                noise_channels=args.noise_channels
            )
            
            loss_meter.update(metrics['loss'], a.shape[0])
            
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.6f}",
                'E_init': f"{metrics['energy_init']:.2f}",
                'E_ref': f"{metrics['energy_refined']:.2f}"
            })
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        train_history.append(loss_meter.avg)
        
        # Validation
        if (epoch + 1) % args.eval_interval == 0:
            print(f"\nRunning validation...")
            val_metrics = evaluate(generator, val_data_precomputed, h, device)
            
            val_history.append((epoch + 1, val_metrics['mean_l2_error']))
            
            print(f"[Epoch {epoch+1}] Val L2 Error: {val_metrics['mean_l2_error']:.6f}, "
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
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_history, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('CD Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
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
    
    print(f"\\nTraining complete. Best L2 error: {best_l2_error:.6f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D Darcy flow solver with Contrastive Divergence")
    
    # Grid
    parser.add_argument('--n_grid', type=int, default=32, help='Interior grid size (N×N)')
    
    # Data
    parser.add_argument('--n_train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=640, help='Number of validation samples')
    parser.add_argument('--a_method', type=str, default='mixed', choices=['grf', 'constant', 'lognormal', 'matern', 'mixed'])
    parser.add_argument('--f_method', type=str, default='constant', choices=['grf', 'sinusoidal', 'point_source', 'constant', 'matern', 'mixed'])
    parser.add_argument('--f_amplitude', type=float, default=20.0, help='Amplitude of forcing term')
    parser.add_argument('--a_min', type=float, default=0.1, help='Minimum permeability value')
    parser.add_argument('--a_max', type=float, default=2.0, help='Maximum permeability value')
    
    # Model
    parser.add_argument('--generator', type=str, default='fno', choices=['cnn', 'mlp', 'fno'])
    parser.add_argument('--noise_channels', type=int, default=8, help='Noise channels for CNN generator')
    parser.add_argument('--base_channels', type=int, default=32, help='Base channels for CNN')
    parser.add_argument('--depth', type=int, default=3, help='Depth of U-Net')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dim for MLP')
    parser.add_argument('--n_layers', type=int, default=4, help='Layers for MLP')
    
    # FNO-specific
    parser.add_argument('--fno_width', type=int, default=32, help='FNO hidden channel width')
    parser.add_argument('--fno_modes', type=int, default=12, help='FNO Fourier modes per dimension')
    parser.add_argument('--fno_layers', type=int, default=4, help='Number of FNO Fourier layers')
    
    # Energy / Langevin
    parser.add_argument('--step_size', type=float, default=20, help='Langevin step size')
    parser.add_argument('--noise_scale', type=float, default=1e-4, help='Langevin noise scale')
    parser.add_argument('--langevin_steps', type=int, default=20, help='Number of Langevin steps')
    
    # Preconditioner
    parser.add_argument('--precond_mode', type=str, default='inverse_laplacian',
                        choices=['matern', 'inverse_laplacian'])
    parser.add_argument('--precond_kappa', type=float, default=1.0, help='Preconditioner κ')
    parser.add_argument('--precond_alpha', type=float, default=1.0, help='Preconditioner α')
    
    # Training
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    # Logging
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--output_dir', type=str, default='./outputs_fno', help='Output directory')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use')
    parser.add_argument('--dry_run', action='store_true', help='Quick test without training')
    
    args = parser.parse_args()
    main(args)
