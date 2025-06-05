# dataset/scattering_dataset.py

import torch
import os
from torch.utils.data import Dataset
import torch.nn.functional as F

class ScatteringDataset(Dataset):
    def __init__(self, data_dir="./generated_dataset"):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path)

        # Es: [B, n_meas, n_inc] → we want to pack real + imag as [2, n_meas, n_inc]
        Es_real = data["Es_real"]  # shape [n_meas, n_inc]
        Es_imag = data["Es_imag"]
        permittivity = data["permittivity"]  # shape [1, H, W]

        # Combine real and imag → shape [2, n_meas, n_inc]
        Es = torch.stack([Es_real, Es_imag], dim=0)  # 2-channel

        return Es, permittivity

