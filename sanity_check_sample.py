import torch
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from models.forward_model import InverseScattering
from dataset.scattering_dataset import ScatteringDataset

def main():
    # Load one .pt sample
    sample_path = "./generated_dataset/sample_0000.pt"
    sample = torch.load(sample_path)

    Es_real = sample['Es_real']       # [n_meas, n_inc]
    Es_imag = sample['Es_imag']
    permittivity = sample['permittivity'].squeeze(0)  # [H, W]

    # Reconstruct full Es
    Es_ref = Es_real + 1j * Es_imag   # [n_meas, n_inc]
    N = permittivity.shape[-1]        # e.g., 40
    print(f"Loaded sample with permittivity shape: {permittivity.shape}, Es shape: {Es_ref.shape}")
    # Plot the ground truth permittivity
    plt.figure(figsize=(5, 4))
    plt.pcolor(permittivity.numpy(), cmap='viridis', shading='auto')
    plt.colorbar(label='Permittivity')
    plt.title("Ground Truth $\\varepsilon_r$")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Instantiate forward model
    model = InverseScattering(image_size=N, n_inc_wave=36, er=1.2)
    model.eval()

    # Recompute Es from permittivity
    with torch.no_grad():
        input_tensor = permittivity.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        Es_pred = model(input_tensor)                          # [1, n_meas, n_inc]
        Es_pred = Es_pred[0]                                   # [n_meas, n_inc]

    # Compute relative error
    rel_error = torch.norm(Es_pred - Es_ref) / torch.norm(Es_ref)
    print(f"Relative error between predicted and saved Es: {rel_error.item():.6f}")

    # Reconstruct using BA
    Es_pred = Es_pred.unsqueeze(0)  # [1, n_meas, n_inc]
    chi_reconstructed = model.BA(Es_pred, gamma=1e-3, image_size=N)  # [1, N, N]

    # Plot the reconstructed image
    plt.figure(figsize=(5, 4))
    plt.pcolor(torch.abs(chi_reconstructed[0]).cpu().numpy(), cmap='viridis', shading='auto')
    plt.colorbar(label='Reconstructed $\\chi$')
    plt.title("BA-Reconstructed Contrast $\\chi$")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

