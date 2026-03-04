"""
Visualize solutions for Poisson equation (-Δu = f).

Generates 4-panel plots for random samples:
1. Forcing field f(x,y)
2. Reference solution u_ref(x,y)
3. Predicted solution u_pred(x,y)
4. Absolute Error |u_pred - u_ref|
"""

import os
import argparse
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator import Generator2d, MLPGenerator2d, FNOGenerator2d
from utils import solve_darcy_exact, compute_relative_l2_error_2d
from data import sample_coefficient_a, sample_forcing_f


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint['args']
    
    # Reconstruct arguments namespace
    args = argparse.Namespace(**args_dict)
    
    # Create generator
    if args.generator == 'cnn':
        generator = Generator2d(
            N=args.n_grid,
            noise_channels=args.noise_channels,
            base_channels=args.base_channels,
            depth=args.depth
        )
    elif args.generator == 'fno':
        generator = FNOGenerator2d(
            N=args.n_grid,
            noise_channels=args.noise_channels,
            width=args.fno_width,
            modes=args.fno_modes,
            n_layers=args.fno_layers
        )
    else:
        # Fallback if mlp
        generator = MLPGenerator2d(
            N=args.n_grid,
            noise_dim=args.noise_channels * args.n_grid**2,
            hidden_dims=[args.hidden_dim] * args.n_layers
        )
        
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator = generator.to(device)
    generator.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} (Best L2: {checkpoint.get('val_metrics', {}).get('mean_l2_error', 'N/A')})")
    return generator, args


def visualize(model_path, n_samples=3, output_path='poisson_viz.png', device='cuda'):
    """Generate visualization."""
    generator, args = load_model(model_path, device)
    N = args.n_grid
    h = 1.0 / (N + 1)
    
    print(f"Generating {n_samples} test samples (a_method={args.a_method}, f_method={args.f_method})...")
    
    # For Poisson, a is usually constant. We use whatever the model was trained with (or force constant)
    # The user asked to "test on Poisson's equation, where we fix a=1".
    # Assuming the loaded model was trained on Poisson or at least compatible.
    # If args.a_method is NOT constant, we can still force constant if desired, 
    # but strictly we should sample from the distribution the model expects.
    # However, if user specifically asks for Poisson visualization, let's sample 'constant' a.
    
    # Force constant a for Poisson visualization as requested
    a = sample_coefficient_a(N, n_samples, method='constant', device=device)
    if args.a_method != 'constant':
        print(f"Warning: Model trained with a_method='{args.a_method}', but visualizing with constant a=1.")
        
    f = sample_forcing_f(N, n_samples, method=args.f_method, 
                       amplitude=args.f_amplitude, 
                       device=device)
    
    # Compute exact solutions
    u_ref = solve_darcy_exact(a, f, h)
    
    # Generate predictions
    noise_channels = args.noise_channels
    xi = torch.randn(n_samples, noise_channels, N, N, device=device)
    
    with torch.no_grad():
        u_pred = generator(a, f, xi)
    
    # Compute errors
    errors = torch.abs(u_pred - u_ref)
    rel_l2 = compute_relative_l2_error_2d(u_pred, u_ref, h)
    
    # Plot
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 4)
        
    for i in range(n_samples):
        # Forcing f
        im0 = axes[i, 0].imshow(f[i].cpu().numpy(), cmap='RdBu_r', origin='lower')
        axes[i, 0].set_title(f"Forcing f\n(Sample {i+1})")
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Reference
        im1 = axes[i, 1].imshow(u_ref[i].cpu().numpy(), cmap='RdBu_r', origin='lower')
        axes[i, 1].set_title("Reference u_ref")
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Predicted
        im2 = axes[i, 2].imshow(u_pred[i].cpu().numpy(), cmap='RdBu_r', origin='lower')
        axes[i, 2].set_title(f"Predicted u_gen\n(Rel L2: {rel_l2[i]:.4f})")
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Error
        im3 = axes[i, 3].imshow(errors[i].cpu().numpy(), cmap='inferno', origin='lower')
        axes[i, 3].set_title(f"Abs Error |u - u_ref|")
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs_fno/darcy2d_run_20260219_015139/best_model.pt', help='Path to model checkpoint (.pt)')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='poisson_viz_fno.png', help='Output filename')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--device', type=str, default=None, help='Specific device (e.g. cuda:2)')
    
    args = parser.parse_args()
    
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    visualize(args.model_path, args.n_samples, args.output, device)
