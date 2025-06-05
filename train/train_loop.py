# train/train_loop.py

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from dataset.scattering_dataset import ScatteringDataset
from models.unet import UNet
import numpy as np

def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_ssim_torch(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return ssim(pred_np, target_np, data_range=1.0)

def visualize_predictions(pred, target, epoch):
    pred_img = pred[0].squeeze().detach().cpu().numpy()
    target_img = target[0].squeeze().cpu().numpy()
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Prediction")
    plt.imshow(pred_img, cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(target_img, cmap='viridis')
    plt.colorbar()
    plt.suptitle(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.show()

def train(
    data_dir="./generated_dataset",
    epochs=200,
    batch_size=8,
    lr=1e-4,
    checkpoint_path="./checkpoints/unet_supervised_initial.pt", # <--- CHANGE THIS LINE
    #checkpoint_path="./checkpoints/unet_latest.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    # Load dataset & split into train/val
    full_dataset = ScatteringDataset(data_dir)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Loaded checkpoint")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Es, perm in train_loader:
            Es, perm = Es.to(device), perm.to(device)
            pred = model(Es)
            #print(f"pred shape: {pred.shape}, perm shape: {perm.shape}")
            loss = loss_fn(pred, perm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ----- Validation -----
        model.eval()
        val_loss, val_psnr, val_ssim = 0, 0, 0
        with torch.no_grad():
            for Es, perm in val_loader:
                Es, perm = Es.to(device), perm.to(device)
                pred = model(Es)
                val_loss += loss_fn(pred, perm).item()
                val_psnr += compute_psnr(pred, perm)
                val_ssim += compute_ssim_torch(pred, perm)

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.3f}")

        # Optional visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_predictions(pred, perm, epoch)

        # Save checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    print("✅ Training complete")

if __name__ == "__main__":
    train()

