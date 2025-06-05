import torch
import torch.nn as nn
import numpy as np
from utils.green_generator import generate_green_matrices  # ⬅️ Import your generator
import matplotlib.pyplot as plt

class InverseScattering(nn.Module):
    def __init__(self, image_size=40, n_inc_wave=36, er=1.1):
        super(InverseScattering, self).__init__()

        # Generate Gs, Gd, Ei dynamically
        Gs, Gd, Ei = generate_green_matrices(
            D=2,
            R=3,
            epsilon_b=1,
            N=image_size,
            Nr=n_inc_wave,
            f=4e8
        )

        # Subsample based on number of incident waves
        Ei = Ei[:, ::(Ei.shape[1] // n_inc_wave)]
        Gs = Gs[::(Gs.shape[0] // n_inc_wave), :]

        # Convert to torch complex tensors
        self.Gd = torch.tensor(Gd, dtype=torch.complex64)
        self.Gs = torch.tensor(Gs, dtype=torch.complex64)
        self.Ei = torch.tensor(Ei, dtype=torch.complex64)

        self.n = n_inc_wave
        self.chai = er - 1
        
        self.image_size = image_size
        print(f"Gd[23,41] is: {Gd[23,41]}")

    def forward(self, x):
        """
        x: torch.Tensor of shape [B, 1, H, W], values in [0, 1]
        Returns: Scattered fields of shape [B, n_meas, n_inc]
        """
        B, _, H, W = x.shape
        #x_scaled = x * (self.chai / 2) + (self.chai / 2)
        x_scaled = x * self.chai  # Scale to [0, chi]
        gt = x_scaled.view(B, -1).to(dtype=self.Gd.dtype)

        I = torch.eye(H * W, dtype=self.Gd.dtype, device=x.device)
        Gd = self.Gd.to(device=x.device)
        Gs = self.Gs.to(device=x.device)
        Ei = self.Ei.to(device=x.device)

        # Batched system matrix inversion
        I_minus_GdGt = I[None, :, :] - Gd[None, :, :] * gt[:, None, :]  # shape [B, HW, HW]
        Et = torch.linalg.solve(I_minus_GdGt, Ei[None, :, :].expand(B, -1, -1))  # [B, HW, n_inc]

        Gsgt = Gs[None, :, :] * gt[:, None, :]  # [B, n_meas, HW]
        Es = torch.bmm(Gsgt, Et)  # [B, n_meas, n_inc]

        return Es
    
    def BA(self, Es, gamma, image_size=40):
        B = Es.shape[0]
        N = image_size
        M = N * N
        n_inc = self.Ei.shape[1]
        n_meas = self.Gs.shape[0]

        device = Es.device
        matrix1 = gamma * torch.eye(M, device=device, dtype=Es.dtype)
    
        # Precompute matrix1 (shared across batch)
        for i in range(n_inc):
            Ge = torch.matmul(self.Gs, torch.diag(self.Ei[:, i]))  # [n_meas, M]
            matrix1 = matrix1 + Ge.conj().T @ Ge  # [M, M]

        # Compute matrix2 for each batch item
        matrix2 = torch.zeros(B, M, 1, dtype=Es.dtype, device=device)

        for i in range(n_inc):
            Ge = torch.matmul(self.Gs, torch.diag(self.Ei[:, i]))  # [n_meas, M]
            Es_i = Es[:, :, i]  # [B, n_meas]
            Ge_H = Ge.conj().T  # [M, n_meas]
            matrix2 += torch.matmul(Ge_H[None, :, :], Es_i[:, :, None])  # [B, M, 1]

        # Solve batched system
        matrix1_exp = matrix1[None, :, :].expand(B, M, M)  # [B, M, M]
        chi_vec = torch.linalg.solve(matrix1_exp, matrix2)  # [B, M, 1]

        chi = chi_vec.view(B, N, N)
        return chi

        dtype = Es.dtype

        J = torch.zeros(B, M, n_inc, dtype=dtype, device=device)
        Et = torch.zeros(B, n_meas, n_inc, dtype=dtype, device=device)

    def BP(self, Es_meas, image_size=None):
        """
        Backpropagation (BP) reconstruction method based on the provided MATLAB code.
        This method is a direct (non-iterative over outer loop) inverse solver.

        Es_meas: Measured scattered fields of shape [B, n_meas, n_inc] (complex)
        gamma: A regularization parameter (not directly used in the current MATLAB-derived formula's loop).
        num_iterations: Not directly applicable to the core MATLAB logic provided (which is a sum).
        image_size: Optional, defaults to self.image_size.
        Returns: Reconstructed permittivity pattern of shape [B, 1, H, W] (real, in range [0,1])
        """
        if image_size is None:
            image_size = self.image_size

        B = Es_meas.shape[0] # Batch size
        N = image_size
        M = N * N # Total number of internal cells
        n_inc = self.Ei.shape[1] # Number of incident waves (Nr in MATLAB)
        n_meas = self.Gs.shape[0] # Number of measurement points (Nr in MATLAB for Rx)

        device = Es_meas.device

        Gs_t = self.Gs.to(device=device) # [n_meas, M]
        Gd_t = self.Gd.to(device=device) # [M, M]
        Ei_t = self.Ei.to(device=device) # [M, n_inc]

        accumulator_a = torch.zeros(B, M, 1, dtype=Es_meas.dtype, device=device)
        accumulator_b = torch.zeros(B, M, 1, dtype=Es_meas.dtype, device=device)

        for p in range(n_inc):
            Es_meas_p = Es_meas[:, :, p] # [B, n_meas]

            # Calculation of gama (step size) per incident wave and per batch sample
            Gs_Gs_H_meas = Gs_t @ Gs_t.conj().T # [n_meas, n_meas]
            term_v = torch.bmm(Gs_Gs_H_meas.expand(B, -1, -1), Es_meas_p.unsqueeze(-1)) # [B, n_meas, 1]

            numerator_gama = torch.bmm(Es_meas_p.unsqueeze(1).conj(), term_v).squeeze(-1) # [B, 1]
            denominator_gama = torch.norm(term_v, dim=1, keepdim=True).pow(2).squeeze(-1) # [B, 1]

            gama_p_batch = numerator_gama / (denominator_gama + 1e-12) # [B, 1]

            # Calculate J(:,p) - Induced Current Density
            J_p_batch = gama_p_batch[:, None, :] * torch.bmm(Gs_t.conj().T.expand(B, -1, -1), Es_meas_p.unsqueeze(-1)) # [B, M, 1]

            # Calculate E_t(:,p) - Total Field
            # FIX: Correctly expand Ei_t[:, p] to [B, M, 1]
            # Ei_t[:, p] is [M], so unsqueeze and expand
            Et_p_batch = Ei_t[:, p].unsqueeze(-1).expand(B, M, 1) + torch.bmm(Gd_t.expand(B, -1, -1), J_p_batch) # [B, M, 1]

            # Accumulate for Khai_BP
            accumulator_a += J_p_batch * Et_p_batch.conj()
            accumulator_b += Et_p_batch.abs().pow(2)

        Khai_BP_vec = accumulator_a / (accumulator_b + 1e-12) # [B, M, 1]

        # The BP method in MATLAB typically reconstructs (epsilon_r - 1).
        # To get a permittivity image in [0,1] range compatible with GT perm:
        # Reconstructed permittivity = real(Khai_BP_vec) / self.chai
        reconstructed_permittivity = Khai_BP_vec.real.view(B, N, N) / self.chai

        # Clamp to ensure output is within [0,1] range for image display/metrics
        reconstructed_permittivity = torch.clamp(reconstructed_permittivity, 0, 1)

        return reconstructed_permittivity.unsqueeze(1) # [B, 1, N, N] for image format


if __name__ == "__main__":
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_size = 32          # Spatial dimension N (e.g., 32x32 image)
    n_incident_waves = 32    # Number of incident waves
    er_object = 1.1          # Relative permittivity of the object
    frequency = 4e8          # Operating frequency

    # --- Instantiate the InverseScattering model (internally generates Gs, Gd, Ei) ---
    scattering_model = InverseScattering(
        image_size=image_size,
        n_inc_wave=n_incident_waves,
        er=er_object
    ).to(device)
    scattering_model.eval()

    # --- Create a Synthetic Object (10x10 square) ---
    synthetic_permittivity = torch.zeros(
        (1, 1, image_size, image_size), dtype=torch.float32, device=device
    )
    # Define square from index 11 to 20 (inclusive), making it 10x10 pixels
    synthetic_permittivity[:, :, 3, 16] = 1.0

    # --- Simulate Scattered Fields (Forward Model) ---
    with torch.no_grad():
        es_simulated_measurements = scattering_model(synthetic_permittivity)

    # --- Compute Reconstruction using Backpropagation (BP) Method ---
    with torch.no_grad():
        reconstructed_permittivity_bp = scattering_model.BP(es_simulated_measurements)
    
    # --- Prepare for Plotting ---
    synthetic_img_np = synthetic_permittivity.squeeze().cpu().numpy()
    reconstructed_img_np = reconstructed_permittivity_bp.squeeze().cpu().numpy()

    # --- Display Result ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4)) # Adjusted figsize for conciseness

    im1 = axes[0].imshow(synthetic_img_np, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off') # Turn off axis labels and ticks
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(reconstructed_img_np*10, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('BP Reconstruction')
    axes[1].axis('off') # Turn off axis labels and ticks
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle(f"BP Reconstruction for {image_size}x{image_size} object with er={er_object}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()
    print(f"reconstructed_img_np[0, 0] is: {reconstructed_img_np[0, 0]}")

