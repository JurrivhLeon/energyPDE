"""
Contrastive Divergence training for the energy-based PDE solver (Nonlinear Poisson).

The generator T_theta(s, xi) is trained to produce samples that match
Langevin-refined samples targeting the Gibbs distribution:
    P*(u|s) ∝ exp(-J(u; s))

The training objective (amortized CD) is:
    L_CD(theta) = E_{s, xi} [ ||T_theta(s, xi) - stopgrad(u_K)||^2 ]
    
where u_K is obtained by running K steps of Langevin dynamics starting from T_theta(s, xi).
"""

import argparse
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from generator import Generator
from energy import compute_energy, compute_energy_gradient, build_laplacian_matrix
from preconditioned_langevin import preconditioned_langevin_refine, MaternPreconditioner
from data import GinzburgLandauDataset
from utils import solve_newton_method, compute_relative_l2_error, AverageMeter, count_parameters
from reference_solver import solve_ginzburg_landau_bvp


def train_step(
    generator: nn.Module,
    s: torch.Tensor,
    h: float,
    step_size: float,
    noise_scale: float,
    langevin_steps: int,
    optimizer: torch.optim.Optimizer,
    noise_dim: int,
    preconditioner: MaternPreconditioner = None,
    K: torch.Tensor = None
) -> dict:
    """
    Perform one training step of contrastive divergence.
    """
    generator.train()
    batch_size = s.shape[0]
    device = s.device
    
    # Sample noise
    xi = torch.randn(batch_size, noise_dim, device=device)
    
    # Generate initial samples
    u0 = generator(s, xi)
    
    # Compute initial energy
    with torch.no_grad():
        energy_init = compute_energy(u0, s, h, K=K).mean()
    
    # Langevin refinement (target samples)
    with torch.no_grad():
        if preconditioner is not None:
            u_refined = preconditioned_langevin_refine(
                u0.detach(), s, h, step_size, noise_scale, langevin_steps, 
                preconditioner, K=K
            )
        else:
            from langevin import langevin_refine
            u_refined = langevin_refine(
                u0.detach(), s, h, step_size, noise_scale, langevin_steps, K=K
            )
        
        
        # Compute refined energy
        energy_refined = compute_energy(u_refined, s, h, K=K).mean()

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
    h: float,
    device: str,
    K: torch.Tensor = None,
    n_samples: int = 1,
) -> dict:
    """
    Evaluate generator by computing mean energy AND L2 error vs reference.
    Expects dataloader to return (s, u_ref).
    """
    generator.eval()
    
    total_l2_error = 0.0
    total_energy_error = 0.0
    total_energy_gap = 0.0
    total_count = 0
    
    with torch.no_grad():
        for s, u_ref in dataloader:
            s = s.to(device)
            u_ref = u_ref.to(device)
            batch_size = s.shape[0]
            
            # Generate samples
            s_expanded = s.repeat_interleave(n_samples, dim=0)
            xi = torch.randn(s_expanded.shape[0], generator.noise_dim, device=device)
            u_gen = generator(s_expanded, xi)
            
            # Expand reference
            u_ref_expanded = u_ref.repeat_interleave(n_samples, dim=0)
            
            # L2 Error
            l2_err = compute_relative_l2_error(u_gen, u_ref_expanded, h)
            total_l2_error += l2_err.sum().item()
            
            # Energy Gap
            energy_gen = compute_energy(u_gen, s_expanded, h, K=K)
            energy_ref = compute_energy(u_ref_expanded, s_expanded, h, K=K)
            total_energy_gap += (energy_gen - energy_ref).abs().sum().item()
            
            total_count += s_expanded.shape[0]
    
    generator.train()
    
    return {
        'mean_l2_error': total_l2_error / total_count,
        'mean_energy_gap': total_energy_gap / total_count
    }


def main(args):
    # Setup
    device = 'cuda:1' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Grid setup
    n_grid = args.n_grid
    h = 1.0 / (n_grid + 1)
    
    # Pre-build stiffness matrix
    K = build_laplacian_matrix(n_grid, h, device=device).to(device)
    
    # Create datasets
    train_dataset = GinzburgLandauDataset(
        n_grid, args.n_train,
        method=args.source_method,
        amplitude=args.source_amplitude,
        length_scale=args.source_length_scale,
        pregenerate=True
    )
    
    val_dataset = GinzburgLandauDataset(
        n_grid, args.n_val,
        method=args.source_method,
        amplitude=args.source_amplitude,
        length_scale=args.source_length_scale,
        pregenerate=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Pre-compute validation solutions
    print(f"\nPre-computing validation reference solutions for {len(val_dataset)} samples...")
    t0 = time.time()
    s_val = val_dataset.s_data.to(device)
    
    # Process in chunks to avoid OOM
    u_ref_list = []
    chunk_size = 100 
    
    with torch.no_grad():
        # Using Scipy BVP reference solver (CPU based)
        # Note: Need to be careful with parallelism if using many workers, but here we do simple loop
        x_full = np.linspace(0, 1, n_grid + 2) # n_grid interior points -> n+2 total
        
        for i in range(0, len(s_val), chunk_size):
            s_batch = s_val[i:i+chunk_size]
            batch_u_ref = []
            
            for j in range(s_batch.shape[0]):
                s_cpu = s_batch[j].cpu().numpy()
                # Pad s for interpolation
                s_full = np.concatenate(([s_cpu[0]], s_cpu, [s_cpu[-1]]))
                
                # Solve using reference BVP
                # Reduce tolerance slightly for speed if needed, but 1e-6 is fine
                u_full, res = solve_ginzburg_landau_bvp(s_full, x_full, tol=1e-5)
                
                if not res.success:
                    # Fallback or warn?
                    # For now just use it but warn
                    # print(f"Warning: Sample k={i+j} BVP failed: {res.message}")
                    pass
                    
                u_interior = u_full[1:-1]
                batch_u_ref.append(u_interior)
                
            batch_u_ref = np.stack(batch_u_ref) # (chunk, n)
            u_ref_list.append(torch.from_numpy(batch_u_ref).float())
            
            if (i // chunk_size) % 5 == 0:
                print(f"  Processed {i}/{len(s_val)} samples with BVP solver")
                
    u_ref_val = torch.cat(u_ref_list, dim=0)
    print(f"Validation pre-computation done in {time.time() - t0:.2f}s")
    
    # Create TensorDataset for validation
    from torch.utils.data import TensorDataset
    val_dataset_full = TensorDataset(val_dataset.s_data, u_ref_val)
    val_loader = DataLoader(val_dataset_full, batch_size=args.batch_size, shuffle=False)
    
    # Create generator
    generator = Generator(
        n_grid=n_grid,
        noise_dim=args.noise_dim,
        hidden_dims=[args.hidden_dim] * args.n_layers,
        activation=args.activation
    ).to(device)
    
    print(f"Generator parameters: {count_parameters(generator):,}")
    
    # Create preconditioner
    preconditioner = MaternPreconditioner(
        n_grid,
        kappa=args.precond_kappa,
        alpha=args.precond_alpha,
        mode='matern',
        device=device
    )
    
    print(f"Preconditioner: Matérn (κ={args.precond_kappa}, α={args.precond_alpha})")
    
    # Optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"cd_run_nl_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining config:")
    print(f"  Grid: {n_grid}")
    print(f"  Langevin: step_size={args.step_size}, noise_scale={args.noise_scale}, K={args.langevin_steps}")
    print(f"  Training samples: {args.n_train}")
    print(f"  Source Method: {args.source_method}")
    print(f"  Output: {output_dir}")
    print()
    
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
        
        for batch_idx, s in enumerate(pbar):
            s = s.to(device)
            
            metrics = train_step(
                generator, s, h, args.step_size,
                args.noise_scale, args.langevin_steps,
                optimizer, args.noise_dim,
                preconditioner=preconditioner,
                K=K
            )
            
            loss_meter.update(metrics['loss'], s.shape[0])
            
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
            val_metrics = evaluate(generator, val_loader, h, device, K=K)
            
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
    parser = argparse.ArgumentParser(description='Train Ginzburg-Landau solver with CD')
    
    # Grid and physics
    parser.add_argument('--n_grid', type=int, default=64, help='Number of interior grid points')
    
    # Data
    parser.add_argument('--n_train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--source_method', type=str, default='mixed', help='Source sampling method (grf, sinusoidal, mixed)')
    parser.add_argument('--source_amplitude', type=float, default=20.0, help='Source amplitude')
    parser.add_argument('--source_length_scale', type=float, default=0.3, help='GRF length scale')
    
    # Generator
    parser.add_argument('--noise_dim', type=int, default=16, help='Noise dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    
    # Langevin
    parser.add_argument('--step_size', type=float, default=1e-4, help='Langevin step size')
    parser.add_argument('--noise_scale', type=float, default=1e-3, help='Langevin noise magnitude')
    parser.add_argument('--langevin_steps', type=int, default=50, help='Number of Langevin steps')
    
    # Preconditioner
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
    
    args = parser.parse_args()
    main(args)
