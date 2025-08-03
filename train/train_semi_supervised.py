# train/train_semi_supervised.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
import os
import numpy as np
import argparse
import time
import random
from itertools import cycle

# --- Import custom modules ---
from dataset.scattering_dataset import ScatteringDataset
from models.forward_model import InverseScattering

# --- Seeding Function for Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- J-Net Architecture ---
class JNet(nn.Module):
    """
    A wrapper for the U-Net that implements the J-Net architecture.
    It takes two inputs (Es and J+) and concatenates them before passing
    them to a standard U-Net.
    """
    def __init__(self, in_channels_es=2, in_channels_j=2, out_channels=1):
        super(JNet, self).__init__()
        # The U-Net will receive the concatenated inputs
        total_in_channels = in_channels_es + in_channels_j
        self.unet = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=total_in_channels,
            out_channels=out_channels,
            init_features=32,
            pretrained=False
        )

    def forward(self, es_input, j_plus_input):
        # Concatenate along the channel dimension (dim=1)
        combined_input = torch.cat([es_input, j_plus_input], dim=1)
        return self.unet(combined_input)

# --- Helper Functions ---
def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def physics_loss_fn(predicted_permittivity, original_Es_stacked, forward_model):
    """
    Calculates the physics-based loss (Maxwell's Loss).
    It checks if the predicted permittivity, when passed through the forward model,
    recreates the original scattered field.
    """
    # Simulate the scattered field from the network's prediction
    simulated_Es = forward_model(predicted_permittivity)
    
    # Compare the simulated field with the original input field
    # We need to stack the real and imaginary parts for comparison
    simulated_Es_stacked = torch.stack([simulated_Es.real, simulated_Es.imag], dim=1)
    
    loss = F.mse_loss(simulated_Es_stacked, original_Es_stacked)
    return loss

def train_ssl(args):
    """
    Main semi-supervised training function.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    # Create one large dataset from the main data directory
    full_dataset = ScatteringDataset(args.data_dir)
    
    # Deterministically split into labeled and unlabeled sets
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    random.Random(args.seed).shuffle(indices)
    
    num_labeled = int(np.floor(args.labeled_percentage * num_samples))
    labeled_indices = indices[:num_labeled]
    unlabeled_indices = indices[num_labeled:]
    
    labeled_dataset = Subset(full_dataset, labeled_indices)
    unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

    print(f"Total samples: {num_samples}")
    print(f"Using {len(labeled_dataset)} samples for supervised loss ({args.labeled_percentage*100}%)")
    print(f"Using {len(unlabeled_dataset)} samples for physics-based loss")

    # Create separate dataloaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True)

    # --- Model, Optimizer, and Loss ---
    model = JNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    supervised_loss_fn = torch.nn.MSELoss()
    
    # --- Forward Model (for J+ and physics loss) ---
    forward_model = InverseScattering(
        image_size=args.forward_model_image_size,
        n_inc_wave=args.n_incident_waves,
        er=args.relative_permittivity_er
    ).to(device).eval()

    # --- Training Loop ---
    print(f"Starting semi-supervised training on {device}...")
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        
        total_supervised_loss = 0
        total_physics_loss = 0
        
        # Use the longer loader to determine the number of steps per epoch
        num_batches = max(len(labeled_loader), len(unlabeled_loader))
        
        # Use cycle to loop over the shorter loader
        train_loader = zip(cycle(labeled_loader), unlabeled_loader) if len(labeled_loader) < len(unlabeled_loader) else zip(labeled_loader, cycle(unlabeled_loader))

        for (labeled_data, unlabeled_data) in train_loader:
            # --- Supervised Step (on labeled data) ---
            es_labeled, perm_labeled = labeled_data
            es_labeled, perm_labeled = es_labeled.to(device), perm_labeled.to(device)
            
            # Calculate J+ for the labeled batch
            j_plus_labeled = forward_model.calculate_J_plus(es_labeled)
            
            # Get model prediction
            pred_perm_labeled = model(es_labeled, j_plus_labeled)
            
            # Compute supervised loss
            sup_loss = supervised_loss_fn(pred_perm_labeled, perm_labeled)
            
            # --- Self-Supervised Step (on unlabeled data) ---
            es_unlabeled, _ = unlabeled_data # We don't use the ground truth here
            es_unlabeled = es_unlabeled.to(device)
            
            # Calculate J+ for the unlabeled batch
            j_plus_unlabeled = forward_model.calculate_J_plus(es_unlabeled)
            
            # Get model prediction
            pred_perm_unlabeled = model(es_unlabeled, j_plus_unlabeled)
            
            # Compute physics-based loss
            phys_loss = physics_loss_fn(pred_perm_unlabeled, es_unlabeled, forward_model)
            
            # --- Combine Losses and Update ---
            total_loss = sup_loss + args.physics_loss_weight * phys_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_supervised_loss += sup_loss.item()
            total_physics_loss += phys_loss.item()

        end_time = time.time()
        epoch_duration = end_time - start_time
        
        avg_sup_loss = total_supervised_loss / num_batches
        avg_phys_loss = total_physics_loss / num_batches
        
        print(f"Epoch {epoch+1}/{args.epochs} | Supervised Loss: {avg_sup_loss:.6f} | Physics Loss: {avg_phys_loss:.6f} | Duration: {epoch_duration:.2f}s")
        
        # (Optional: Add a validation loop here to monitor PSNR/SSIM on a hold-out set)

        # --- Save Checkpoint ---
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised Training Script for JP-Net")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the full dataset.')
    parser.add_argument('--labeled_percentage', type=float, default=0.2, help='Percentage of the dataset to use as labeled data (e.g., 0.2 for 20%).')
    parser.add_argument('--physics_loss_weight', type=float, default=1.0, help='Weighting factor for the physics-based loss.')
    
    # Standard training arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    
    # Forward model arguments
    parser.add_argument('--forward_model_image_size', type=int, default=32)
    parser.add_argument('--n_incident_waves', type=int, default=32)
    parser.add_argument('--relative_permittivity_er', type=float, required=True)
    
    args = parser.parse_args()
    train_ssl(args)
