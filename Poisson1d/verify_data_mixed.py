
import torch
import matplotlib.pyplot as plt
import numpy as np
from data import sample_forcing_f

def verify_mixed_sampling():
    n_interior = 50
    n_samples = 20
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Sampling 20 mixed forcing terms...")
    f = sample_forcing_f(n_interior, n_samples, method='mixed', amplitude=10.0)
    
    print(f"Shape: {f.shape}")
    print(f"Mean amplitude: {f.abs().max(dim=1)[0].mean()}")
    
    # Plot
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()
    
    x = torch.linspace(0, 1, n_interior)
    
    for i in range(n_samples):
        axes[i].plot(x, f[i].numpy())
        axes[i].set_title(f"Sample {i+1}")
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('mixed_forcing_samples.png')
    print("Saved mixed_forcing_samples.png")

if __name__ == "__main__":
    verify_mixed_sampling()
