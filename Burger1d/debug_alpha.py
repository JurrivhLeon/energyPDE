
import torch
import numpy as np
import matplotlib.pyplot as plt
from preconditioned_langevin import FourierPreconditioner

def test_alpha_effect(n=128, L=1.0, kappa=1.0):
    """
    Compare effective stiffness and noise amplitudes for alpha=1.0 and 2.0.
    """
    print(f"Analyzing alpha effect for n={n}, L={L}, kappa={kappa}")
    
    # Grid of wavenumbers (k integers)
    k_int = torch.fft.fftfreq(n, d=1.0) * n
    k_sq = k_int ** 2
    
    # 1. Physics Stiffness (Diffusion term ~ -k^2, Energy ~ k^4)
    # The restoring force (gradient of energy) scales as k^4 * u_hat
    stiffness_phys = k_sq ** 2
    
    alphas = [1.0, 1.5, 2.0]
    
    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")
        # Eigenvalues
        # (kappa^2 + k^2)^(-alpha)
        # Normalized max=1
        evals = (kappa**2 + k_sq) ** (-alpha)
        evals = evals / evals.max()
        
        # Effective Stiffness (M * Stiffness_phys)
        # This determines how strongly high modes are pulled back to 0
        eff_stiffness = evals * stiffness_phys
        
        # Noise Amplitude (M^{1/2} * xi)
        # xi has variance 1. M^{1/2} scales it.
        noise_amp = evals.sqrt()
        
        # Signal-to-Noise Ratio (Restoring force / Noise) at high k
        # Proportional to eff_stiffness / noise_amp
        # = (evals * k^4) / evals^0.5 = evals^0.5 * k^4
        snr = noise_amp * stiffness_phys
        
        # Look at Nyquist frequency (max k = n/2)
        k_nyq = n/2
        k_mid = n/4
        
        def get_val(arr, k_target):
            # approximate index
            idx = int(k_target)
            return arr[idx].item()
        
        val_stiff = get_val(eff_stiffness, k_nyq)
        val_noise = get_val(noise_amp, k_nyq)
        
        print(f"  At Nyquist (k={k_nyq}):")
        print(f"    Restoring Force factor (M*k^4): {val_stiff:.2e}")
        print(f"    Noise Amplitude (M^0.5):        {val_noise:.2e}")
        print(f"    Ratio (Restoring/Noise):        {val_stiff/val_noise:.2e}")
        
if __name__ == "__main__":
    test_alpha_effect()
