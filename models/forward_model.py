import torch
import torch.nn as nn
import numpy as np
from utils.green_generator import generate_green_matrices  # ⬅️ Import your generator
import matplotlib.pyplot as plt

class InverseScattering(nn.Module):
    def __init__(self, image_size=32, n_inc_wave=32, er=1.1):
        super(InverseScattering, self).__init__()

        Gs, Gd, Ei = generate_green_matrices(
            D=2, R=3, epsilon_b=1, N=image_size, Nr=n_inc_wave, f=4e8
        )
        Ei = Ei[:, ::(Ei.shape[1] // n_inc_wave)]
        Gs = Gs[::(Gs.shape[0] // n_inc_wave), :]

        self.Gd = torch.tensor(Gd, dtype=torch.complex64)
        self.Gs = torch.tensor(Gs, dtype=torch.complex64)
        self.Ei = torch.tensor(Ei, dtype=torch.complex64)

        self.n = n_inc_wave
        self.chai = er - 1
        self.image_size = image_size
        
        U, S, Vh = torch.linalg.svd(self.Gs, full_matrices=False)
        self.U = U.to(dtype=torch.complex64)
        self.S = S.to(dtype=torch.complex64)
        self.Vh = Vh.to(dtype=torch.complex64)
        print("SVD of Gs matrix pre-computed for J-Net.")

        # --- FIX: Pre-compute the inverse of Gd for efficiency ---
        print("Pre-computing inverse of Gd matrix...")
        self.Gd_inv = torch.linalg.inv(self.Gd)
        print("Inverse of Gd pre-computed.")


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

    def get_J_plus_vector(self, Es_stacked):
        """
        Calculates the radiating component of the current density (J+) and
        returns it as a vector, which is needed for verification.
        """
        B, _, n_meas, n_inc = Es_stacked.shape
        device = Es_stacked.device

        Es = Es_stacked[:, 0, :, :] + 1j * Es_stacked[:, 1, :, :]

        U = self.U.to(device)
        S = self.S.to(device)
        V = self.Vh.mH.to(device)

        S_inv = torch.diag(1.0 / S)
        Uh_Es = torch.matmul(U.mH.unsqueeze(0), Es)
        J_plus = torch.matmul(S_inv, Uh_Es)
        J_plus_vector = torch.matmul(V, J_plus) # Shape: [B, M, n_inc]
        
        return J_plus_vector

    def calculate_J_plus(self, Es_stacked):
        """
        Calculates the radiating component of the current density (J+)
        and returns it as a 2-channel image for input to the J-Net.
        """
        B, _, n_meas, n_inc = Es_stacked.shape
        
        J_plus_vector = self.get_J_plus_vector(Es_stacked) # Shape: [B, M, n_inc]
        
        J_plus_avg = J_plus_vector.mean(dim=2) # Shape: [B, M]
        J_plus_img = J_plus_avg.view(B, self.image_size, self.image_size)
        J_plus_stacked = torch.stack([J_plus_img.real, J_plus_img.imag], dim=1)

        return J_plus_stacked

    def reconstruct_chi_from_output(self, network_output_stacked, J_plus_vector):
        """
        Calculates the final permittivity (chi) from the J-Net's output.
        This implements the analytical reconstruction steps from the paper.

        Args:
            network_output_stacked (torch.Tensor): The direct output of the J-Net,
                representing Gd*J-. Shape: [B, 2, H, W].
            J_plus_vector (torch.Tensor): The radiating current vector.
                Shape: [B, M, n_inc].
        """
        device = network_output_stacked.device
        B, _, H, W = network_output_stacked.shape
        M = H * W

        # Convert the network's 2-channel real/imag output to a complex vector
        E_minus_predicted = network_output_stacked[:, 0, :, :] + 1j * network_output_stacked[:, 1, :, :]
        E_minus_predicted_vec = E_minus_predicted.view(B, M, 1)

        # Move physics matrices to the correct device
        Ei = self.Ei.to(device)
        Gd = self.Gd.to(device)
        Gd_inv = self.Gd_inv.to(device)

        # Average over incident waves for a single reconstruction
        J_plus_avg_vec = J_plus_vector.mean(dim=2, keepdim=True) # Shape: [B, M, 1]
        Ei_avg_vec = Ei.mean(dim=1, keepdim=True).unsqueeze(0).expand(B, -1, -1) # Shape: [B, M, 1]

        # 1. Calculate E_total = Ei + Gd*J+ + (Network Output)
        E_plus = torch.matmul(Gd, J_plus_avg_vec)
        E_total_vec = Ei_avg_vec + E_plus + E_minus_predicted_vec

        # 2. Calculate J_total = inv(Gd) * (E_total - Ei)
        J_total_vec = torch.matmul(Gd_inv, E_total_vec - Ei_avg_vec)

        # 3. Calculate chi = J_total / E_total (with stabilization)
        chi_vec = J_total_vec / (E_total_vec + 1e-8)

        # Reshape to image format and take the real part for the final image
        chi_img = chi_vec.real.view(B, 1, H, W)

        # Clamp to the expected [0, 1] normalized range
        return torch.clamp(chi_img, 0, 1)


    
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

    def BP(self, Es_meas_stacked, image_size=None):
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

        B = Es_meas_stacked.shape[0] # Batch size
        N = image_size
        M = N * N # Total number of internal cells
        n_inc = self.Ei.shape[1] # Number of incident waves (Nr in MATLAB)
        n_meas = self.Gs.shape[0] # Number of measurement points (Nr in MATLAB for Rx)

        device = Es_meas_stacked.device

        Gs_t = self.Gs.to(device=device) # [n_meas, M]
        Gd_t = self.Gd.to(device=device) # [M, M]
        Ei_t = self.Ei.to(device=device) # [M, n_inc]

        # NEW: Convert Es_meas_stacked [B, 2, n_meas, n_inc] to complex Es_meas [B, n_meas, n_inc]
        Es_meas_real = Es_meas_stacked[:, 0, :, :] # [B, n_meas, n_inc]
        Es_meas_imag = Es_meas_stacked[:, 1, :, :] # [B, n_meas, n_inc]
        Es_meas = Es_meas_real + 1j * Es_meas_imag # [B, n_meas, n_inc] (complex)

        accumulator_a = torch.zeros(B, M, 1, dtype=Es_meas.dtype, device=device)
        accumulator_b = torch.zeros(B, M, 1, dtype=Es_meas.dtype, device=device)

        for p in range(n_inc):
            Es_meas_p = Es_meas[:, :, p] # [B, n_meas]

            # Calculation of gama (step size) per incident wave and per batch sample
            Gs_Gs_H_meas = Gs_t @ Gs_t.conj().T # [n_meas, n_meas]
            term_v = torch.bmm(Gs_Gs_H_meas.expand(B, -1, -1), Es_meas_p.unsqueeze(-1)) # [B, n_meas, 1]

            numerator_gama = torch.bmm(Es_meas_p.unsqueeze(1).conj(), term_v).squeeze(-1) # [B, 1]
            denominator_gama = torch.norm(term_v, dim=1, keepdim=True).pow(2).squeeze(-1) # [B, 1]

            gama_p_batch = numerator_gama / (denominator_gama) # [B, 1]

            # Calculate J(:,p) - Induced Current Density
            J_p_batch = gama_p_batch[:, None, :] * torch.bmm(Gs_t.conj().T.expand(B, -1, -1), Es_meas_p.unsqueeze(-1)) # [B, M, 1]

            # Calculate E_t(:,p) - Total Field
            # FIX: Correctly expand Ei_t[:, p] to [B, M, 1]
            # Ei_t[:, p] is [M], so unsqueeze and expand
            Et_p_batch = Ei_t[:, p].unsqueeze(-1).expand(B, M, 1) + torch.bmm(Gd_t.expand(B, -1, -1), J_p_batch) # [B, M, 1]

            # Accumulate for Khai_BP
            accumulator_a += J_p_batch * Et_p_batch.conj()
            accumulator_b += Et_p_batch.abs().pow(2)

        Khai_BP_vec = accumulator_a / (accumulator_b) # [B, M, 1]

        # The BP method in MATLAB typically reconstructs (epsilon_r - 1).
        # To get a permittivity image in [0,1] range compatible with GT perm:
        # Reconstructed permittivity = real(Khai_BP_vec) / self.chai
        reconstructed_permittivity = Khai_BP_vec.real.view(B, N, N)

        # Clamp to ensure output is within [0,1] range for image display/metrics
        reconstructed_permittivity = torch.clamp(reconstructed_permittivity, 0, 1)

        return reconstructed_permittivity.unsqueeze(1) # [B, 1, N, N] for image format
