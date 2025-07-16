# train/train_supervised.py (Final)

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import argparse
import time
import random

# --- Import custom modules ---
from dataset.scattering_dataset import ScatteringDataset
from models.forward_model import InverseScattering
from models.dncnn import DnCNN
from models.swinir import SwinIR

# --- Seeding Function for Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Model Loader ---
def get_model(model_name, in_channels, out_channels, image_size):
    """
    Loads and returns the specified model.
    """
    print(f"Loading model: {model_name}")
    if model_name.lower() == 'unet':
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=32,
            pretrained=False
        )
    elif model_name.lower() == 'dncnn':
        model = DnCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=17,
            features=64
        )
    elif model_name.lower() == 'swinir':
        model = SwinIR(
            img_size=image_size,
            in_chans=in_channels,
            out_chans=out_channels,
            embed_dim=96,
            depths=[6, 6, 6, 6],
            num_heads=[6, 6, 6, 6],
            window_size=8,
            mlp_ratio=4.,
            upscale=1,
            upsampler='',
        )
    else:
        raise ValueError(f"Model '{model_name}' not recognized or implemented.")
        
    return model

# --- Helper Functions ---
def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def compute_ssim_torch(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    data_range = target_np.max() - target_np.min()
    if data_range == 0: return 1.0
    return ssim(pred_np, target_np, data_range=data_range, channel_axis=None)

def train(args):
    """
    Main training function.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    full_dataset = ScatteringDataset(args.data_dir)
    generator = torch.Generator().manual_seed(args.seed)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Model Selection ---
    in_channels = 1 if args.input_type == "BP" else 2
    model = get_model(
        model_name=args.model_name,
        in_channels=in_channels,
        out_channels=1,
        image_size=args.forward_model_image_size
    ).to(device)

    # --- Optimizer and Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    
    # --- Forward Model (for BP input type) ---
    forward_model = None
    if args.input_type == 'BP':
        forward_model = InverseScattering(
            image_size=args.forward_model_image_size,
            n_inc_wave=args.n_incident_waves,
            er=args.relative_permittivity_er
        ).to(device).eval()

    # --- Checkpoint Loading ---
    if os.path.exists(args.checkpoint_path):
        print(f"Resuming training from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(f"Starting new training for {args.model_name}. No checkpoint found.")

    # --- Initial Evaluation Before Training ---
    print("\n--- Evaluating initial model performance ---")
    model.eval()
    initial_val_loss, initial_val_psnr, initial_val_ssim = 0, 0, 0
    with torch.no_grad():
        for Es_gt, perm_gt in val_loader:
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)
            if args.input_type == 'BP':
                net_input = forward_model.BP(Es_gt)
            else:
                net_input = Es_gt
            
            pred_permittivity = model(net_input)
            initial_val_loss += loss_fn(pred_permittivity, perm_gt).item()
            initial_val_psnr += compute_psnr(pred_permittivity, perm_gt)
            initial_val_ssim += compute_ssim_torch(pred_permittivity, perm_gt)
    
    avg_initial_loss = initial_val_loss / len(val_loader)
    avg_initial_psnr = initial_val_psnr / len(val_loader)
    avg_initial_ssim = initial_val_ssim / len(val_loader)
    print(f"Initial Val Loss: {avg_initial_loss:.6f} | Initial Val PSNR: {avg_initial_psnr:.2f} | Initial Val SSIM: {avg_initial_ssim:.4f}")
    print("--------------------------------------------")


    # --- Training Loop ---
    print(f"\nStarting training on {device}...")
    for epoch in range(args.epochs):
        start_time = time.time() # Start timer for the epoch
        model.train()
        total_train_loss = 0
        for Es_gt, perm_gt in train_loader:
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)
            
            if args.input_type == 'BP':
                net_input = forward_model.BP(Es_gt)
            else:
                net_input = Es_gt
            
            pred_permittivity = model(net_input)
            loss = loss_fn(pred_permittivity, perm_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- Validation ---
        model.eval()
        total_val_loss, total_val_psnr, total_val_ssim = 0, 0, 0
        with torch.no_grad():
            for Es_gt, perm_gt in val_loader:
                Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)
                if args.input_type == 'BP':
                    net_input = forward_model.BP(Es_gt)
                else:
                    net_input = Es_gt
                
                pred_permittivity = model(net_input)
                total_val_loss += loss_fn(pred_permittivity, perm_gt).item()
                total_val_psnr += compute_psnr(pred_permittivity, perm_gt)
                total_val_ssim += compute_ssim_torch(pred_permittivity, perm_gt)

        end_time = time.time() # End timer for the epoch
        epoch_duration = end_time - start_time

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_psnr = total_val_psnr / len(val_loader)
        avg_val_ssim = total_val_ssim / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val PSNR: {avg_val_psnr:.2f} | Val SSIM: {avg_val_ssim:.4f} | Duration: {epoch_duration:.2f}s")

        # --- Save Checkpoint ---
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Training Script")
    parser.add_argument('--model_name', type=str, default='unet', choices=['unet', 'dncnn', 'swinir'], help='Name of the model architecture to use.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_type', type=str, required=True, choices=['BP', 'Es'])
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--forward_model_image_size', type=int, default=32)
    parser.add_argument('--n_incident_waves', type=int, default=32)
    parser.add_argument('--relative_permittivity_er', type=float, required=True)
    
    args = parser.parse_args()
    train(args)
