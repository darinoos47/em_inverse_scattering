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
