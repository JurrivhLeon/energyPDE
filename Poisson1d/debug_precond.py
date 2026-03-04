
import torch
import numpy as np
import matplotlib.pyplot as plt
from preconditioned_langevin import MaternPreconditioner, dst_type1, idst_type1

def test_consistency():
    n_grid = 64
    kappa = 1.0
    alpha = 2.0
    
    # CPU Preconditioner
    precond_cpu = MaternPreconditioner(n_grid, kappa, alpha, normalize=True, device='cpu')
    
    # CUDA Preconditioner (if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    precond_gpu = MaternPreconditioner(n_grid, kappa, alpha, normalize=True, device=device)
    
    print(f"Testing Consistency (alpha={alpha})...")
    
    # 1. Compare Eigenvalues
    eig_cpu = precond_cpu.eigenvalues.numpy()
    eig_gpu = precond_gpu.eigenvalues.cpu().numpy()
    eig_diff = np.abs(eig_cpu - eig_gpu).max()
    print(f"Max Eigenvalue Diff (CPU vs {device}): {eig_diff:.4e}")
    
    # 2. DST Consistency
    x = torch.randn(1, n_grid, device='cpu')
    y_cpu = dst_type1(x)
    y_gpu = dst_type1(x.to(device)).cpu()
    dst_diff = (y_cpu - y_gpu).abs().max().item()
    print(f"DST Diff: {dst_diff:.4e}")
    
    # 3. Noise Pattern Visual Check
    # We fix seed and sample
    torch.manual_seed(42)
    noise_cpu = precond_cpu.sample_noise((1, n_grid))
    
    torch.manual_seed(42)
    noise_gpu = precond_gpu.sample_noise((1, n_grid)).cpu()
    
    noise_diff = (noise_cpu - noise_gpu).abs().max().item()
    print(f"Noise Sample Diff (Fixed Seed): {noise_diff:.4e} (Expected 0 if same device, maybe >0 if different backend)")
    
    # 4. Check Spectral Decay
    # Theoretically lambda_m ~ 1 / m^(2*alpha)
    m = np.arange(1, n_grid + 1)
    lambda_m = precond_cpu.eigenvalues.numpy()
    
    # For alpha=2.0, lambda_4 / lambda_1 should be approx (1/4)^4 = 1/256
    ratio_theory = (1.0/4.0)**(2*alpha)
    ratio_actual = lambda_m[3] / lambda_m[0]
    print(f"Eigenvalue Ratio lambda_4/lambda_1:")
    print(f"  Theory (approx): {ratio_theory:.4e}")
    print(f"  Actual:          {ratio_actual:.4e}")
    
    # 5. Effective Step Size for Mode 4
    eta = 0.005
    eff_step = eta * ratio_actual
    print(f"Effective Step Size for Mode 4 (eta={eta}): {eff_step:.4e}")
    print(f"Number of steps to move mode 4 by order 1: {1.0/eff_step:.1f}")

    # Plot noise samples to see "pattern"
    plt.figure(figsize=(10, 5))
    plt.plot(noise_cpu[0].numpy(), label='CPU Sample')
    plt.plot(noise_gpu[0].numpy(), label='GPU Sample')
    plt.title(f'Noise Comparison (alpha={alpha})')
    plt.legend()
    plt.savefig('debug_noise_compare.png')
    print("Saved debug_noise_compare.png")

if __name__ == "__main__":
    test_consistency()
