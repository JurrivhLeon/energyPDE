"""
Visualization script for 1D Nonlinear Poisson (Ginzburg-Landau) equation results.

Loads a trained model and compares generated solutions with exact solutions
computed via the high-precision BVP reference solver.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from generator import Generator
from data import sample_source_s
from utils import get_grid_points, compute_relative_l2_error
from reference_solver import solve_ginzburg_landau_bvp

def visualize_comparison(
    model_path: str,
    n_samples: int = 3,
    device: str = 'cpu',
    save_path: str = 'vis_results/comparison_nl.png'
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
    generator = Generator(
        n_grid=n_interior,
        noise_dim=args['noise_dim'],
        hidden_dims=[args['hidden_dim']] * args['n_layers'],
        activation=args.get('activation', 'gelu')
    )
    
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.to(device)
    generator.eval()
    
    # Generate test samples
    print("Generating test samples...")
    
    # 1. In-Distribution (Same method as training)
    train_method = args.get('source_method', 'mixed')
    train_amp = args.get('source_amplitude', 20.0)
    train_ls = args.get('source_length_scale', 0.3)
    
    s_in = sample_source_s(
        n_interior, n_samples, 
        method=train_method, 
        amplitude=train_amp,
        length_scale=train_ls,
        device=device
    )
    
    # 2. OOD Samples
    # High Frequency Sinusoidal
    s_ood1 = sample_source_s(
        n_interior, 2, 
        method='sinusoidal', 
        amplitude=train_amp * 1.5,
        device=device
    )
    # Mixed / Different GRF
    s_ood2 = sample_source_s(
        n_interior, 2, 
        method='grf', 
        amplitude=train_amp, 
        length_scale=0.1, # Higher freq
        device=device
    )
    
    s_all = torch.cat([s_in, s_ood1, s_ood2], dim=0)
    
    labels = [f"In-Dist ({train_method})"] * n_samples + \
             ["OOD (High Freq Sin)"] * 2 + \
             ["OOD (High Freq GRF)"] * 2
             
    total_samples = len(labels)
    
    # Compute Model Predictions
    print("Computing model predictions...")
    with torch.no_grad():
        # Sample noise
        xi = torch.randn(total_samples, generator.noise_dim, device=device)
        u_pred = generator(s_all, xi)
        
    # Compute Exact Solutions via BVP Reference Solver
    print("Computing exact solutions via BVP solver...")
    u_exact_list = []
    x_full = np.linspace(0, 1, n_interior + 2)
    s_all_cpu = s_all.cpu().numpy()
    
    for i in range(total_samples):
        s_i = s_all_cpu[i]
        # Pad for interpolation
        s_full = np.concatenate(([s_i[0]], s_i, [s_i[-1]]))
        
        u_bvp, res = solve_ginzburg_landau_bvp(s_full, x_full, tol=1e-6)
        
        if not res.success:
            print(f"Warning: Sample {i} BVP solver failed: {res.message}")
            
        u_interior = u_bvp[1:-1]
        u_exact_list.append(u_interior)
        
    u_exact = torch.tensor(np.stack(u_exact_list), dtype=torch.float32, device=device)
    
    # Compute Errors
    l2_errors = compute_relative_l2_error(u_pred, u_exact, h)
    
    # Plotting
    print("Plotting results...")
    x_grid = x_full # [0, ..., 1]
    x_int = x_full[1:-1]
    
    fig, axes = plt.subplots(total_samples, 2, figsize=(12, 3.5 * total_samples))
    
    if total_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(total_samples):
        # 1. Source s(x)
        ax = axes[i, 0]
        ax.plot(x_int, s_all[i].cpu().numpy(), 'g-', label='s(x)')
        ax.set_title(f"Sample {i+1} [{labels[i]}]: Source s(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Solution u(x)
        ax = axes[i, 1]
        u_ex_plot = np.concatenate(([0], u_exact[i].cpu().numpy(), [0]))
        u_pr_plot = np.concatenate(([0], u_pred[i].cpu().numpy(), [0]))
        
        ax.plot(x_grid, u_ex_plot, 'k-', linewidth=2, label='Exact (BVP)')
        ax.plot(x_grid, u_pr_plot, 'r--', linewidth=2, label='Predicted (Gen)')
        
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
    parser.add_argument('--model_path', type=str, default='outputs/cd_run_nl_20260122_010703/best_model.pt', help='Path to trained model checkpoint')
    parser.add_argument('--n_samples', type=int, default=6, help='Number of In-Dist samples')
    parser.add_argument('--output', type=str, default='vis_results/comparison_nl.png', help='Output filename')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    visualize_comparison(args.model_path, args.n_samples, device, args.output)
