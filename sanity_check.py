# sanity_check.py

import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from dataset.scattering_dataset import ScatteringDataset

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = ScatteringDataset(data_dir="./generated_dataset")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Load model
    model = UNet(in_channels=2, out_channels=1).to(device)
    model.eval()

    # Run a forward pass on one batch
    for Es, perm in dataloader:
        print(f"Original Es shape: {Es.shape}")  # Should be [B, 2, 36, 36]
        print(f"Permittivity shape: {perm.shape}")  # Should be [B, 1, H, W] — e.g., [4, 1, 32, 32]

        Es = Es.to(device)
        perm = perm.to(device)

        with torch.no_grad():
            out = model(Es)

        print(f"Output shape: {out.shape}")  # Should be [B, 1, 32, 32]
        break

if __name__ == "__main__":
    main()

