"""
Visualization script for 1D Poisson equation results.

Loads a trained model and compares generated solutions with exact solutions.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from generator import Generator, ConditionalGenerator
from data import sample_coefficient_a, sample_forcing_f
from utils import solve_poisson_exact, compute_relative_l2_error, get_grid_points

def visualize_comparison(
    model_path: str,
    n_samples: int = 3,
    device: str = 'cpu',
    save_path: str = 'vis_results/comparison.png'
):
    """
    Visualize model predictions vs exact solutions.
    """
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint['args']
    
    # Grid setup
    n_interior = args['n_grid']
    h = 1.0 / (n_interior + 1)
    
    # Initialize model
    if args['generator'] == 'simple':
        generator = Generator(
            n_grid=n_interior,
            noise_dim=args['noise_dim'],
            hidden_dims=[args['hidden_dim']] * args['n_layers']
        )
    else:
        generator = ConditionalGenerator(
            n_grid=n_interior,
            noise_dim=args['noise_dim']
        )
    
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.to(device)
    generator.eval()
    
    # Generate test samples
    # We use the same parameters as in training
    # Generate test samples (In-Distribution)
    a_in = sample_coefficient_a(
        n_interior, n_samples, 
        method=args['a_method'], 
        a_min=args['a_min'], 
        a_max=args['a_max'],
        device=device
    )
    
    # Ensure backward compatibility if f_amplitude missing
    f_amp = args.get('f_amplitude', 15.0)
    
    f_in = sample_forcing_f(
        n_interior, n_samples, 
        method='mixed', 
        amplitude=f_amp,
        device=device
    )
    
    # OOD Samples: 2 Trigonometric, 2 Polynomial
    # We maintain the same coefficient distribution
    a_ood = sample_coefficient_a(
        n_interior, 4, 
        method=args['a_method'], 
        a_min=args['a_min'], 
        a_max=args['a_max'],
        device=device
    )
    
    f_sin = sample_forcing_f(n_interior, 2, method='sinusoidal', amplitude=f_amp, device=device)
    f_poly = sample_forcing_f(n_interior, 2, method='polynomial', amplitude=f_amp, device=device)
    f_ood = torch.cat([f_sin, f_poly], dim=0)
    
    # Combine all
    a_all = torch.cat([a_in, a_ood], dim=0)
    f_all = torch.cat([f_in, f_ood], dim=0)
    
    labels = [f"In-Dist ({args['f_method']})"] * n_samples + ["OOD (Sinusoidal)"] * 2 + ["OOD (Polynomial)"] * 2
    total_samples = len(labels)
    
    # Compute exact solutions
    u_exact = solve_poisson_exact(a_all, f_all, h)
    
    # Model predictions
    with torch.no_grad():
        u_pred = generator(a_all, f_all)
        l2_errors = compute_relative_l2_error(u_pred, u_exact, h)
    
    # Plotting
    x_full = get_grid_points(n_interior, include_boundary=True).cpu().numpy()
    x_int = get_grid_points(n_interior, include_boundary=False).cpu().numpy()
    
    fig, axes = plt.subplots(total_samples, 3, figsize=(15, 3.5 * total_samples))
    
    if total_samples == 1:
        axes = np.expand_dims(axes, 0)
    
    for i in range(total_samples):
        # 1. Coefficient a(x)
        ax = axes[i, 0]
        ax.plot(x_full, a_all[i].cpu().numpy(), 'g-', label='a(x)')
        ax.set_title(f"Sample {i+1} [{labels[i]}]: Coefficient a(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Forcing f(x)
        ax = axes[i, 1]
        ax.plot(x_int, f_all[i].cpu().numpy(), 'm-', label='f(x)')
        ax.set_title(f"Sample {i+1} [{labels[i]}]: Forcing f(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Solution u(x)
        ax = axes[i, 2]
        
        # Add boundary zeros for plotting
        u_ex_plot = np.concatenate(([0], u_exact[i].cpu().numpy(), [0]))
        u_pr_plot = np.concatenate(([0], u_pred[i].cpu().numpy(), [0]))
        
        ax.plot(x_full, u_ex_plot, 'k-', linewidth=2, label='Exact')
        ax.plot(x_full, u_pr_plot, 'r--', linewidth=2, label='Predicted')
        
        error = l2_errors[i].item()
        ax.set_title(f"Solution u(x) (Rel L2 Error: {error:.4f})")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")
    
    print("\nSample Errors:")
    for i in range(total_samples):
        print(f"  Sample {i+1} [{labels[i]}]: Rel L2 Error = {l2_errors[i].item():.6f}")
    print(f"  Mean Error: {l2_errors.mean().item():.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs/cd_run_20260121_143926/best_model.pt', help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--n_samples', type=int, default=6, help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='vis_results/comparison.png', help='Output filename')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    visualize_comparison(args.model_path, args.n_samples, device, args.output)
