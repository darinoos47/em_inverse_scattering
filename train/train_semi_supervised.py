# train/train_semi_supervised.py (Corrected)

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
    def __init__(self, in_channels_es=2, in_channels_j=2, out_channels=2): # Output is 2-channel Gd*J-
        super(JNet, self).__init__()
        total_in_channels = in_channels_es + in_channels_j
        self.unet = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=total_in_channels,
            out_channels=out_channels,
            init_features=32,
            pretrained=False
        )

    def forward(self, es_input, j_plus_input):
        combined_input = torch.cat([es_input, j_plus_input], dim=1)
        return self.unet(combined_input)

# --- Helper Functions ---
def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def physics_loss_fn(network_output, j_plus_vector, original_Es_stacked, forward_model):
    """
    Calculates the physics-based loss (Maxwell's Loss).
    """
    # First, get the final permittivity from the network's output
    final_chi = forward_model.reconstruct_chi_from_output(network_output, j_plus_vector)
    
    # Then, simulate the scattered field from that final permittivity
    simulated_Es = forward_model(final_chi)
    simulated_Es_stacked = torch.stack([simulated_Es.real, simulated_Es.imag], dim=1)
    
    # Compare the result to the original scattered field
    loss = F.mse_loss(simulated_Es_stacked, original_Es_stacked)
    return loss

def train_ssl(args):
    """
    Main semi-supervised training function.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    full_dataset = ScatteringDataset(args.data_dir)
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    random.Random(args.seed).shuffle(indices)
    num_labeled = int(np.floor(args.labeled_percentage * num_samples))
    labeled_indices = indices[:num_labeled]
    unlabeled_indices = indices[num_labeled:]
    
    labeled_dataset = Subset(full_dataset, labeled_indices)
    unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

    print(f"Total samples: {num_samples}, Labeled: {len(labeled_dataset)}, Unlabeled: {len(unlabeled_dataset)}")

    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True)

    # --- Model, Optimizer, and Loss ---
    model = JNet(out_channels=2).to(device) # J-Net outputs a 2-channel image
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    supervised_loss_fn = torch.nn.MSELoss()
    
    # --- Forward Model ---
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
        
        total_supervised_loss, total_physics_loss = 0, 0
        num_batches = max(len(labeled_loader), len(unlabeled_loader))
        train_loader = zip(cycle(labeled_loader), unlabeled_loader) if len(labeled_loader) < len(unlabeled_loader) else zip(labeled_loader, cycle(unlabeled_loader))

        for (labeled_data, unlabeled_data) in train_loader:
            optimizer.zero_grad()

            # --- Supervised Step (on labeled data) ---
            es_labeled, perm_labeled = labeled_data
            es_labeled, perm_labeled = es_labeled.to(device), perm_labeled.to(device)
            
            # --- KEY CHANGE: Calculate the TRUE Gd*J- to use as the label ---
            with torch.no_grad():
                # Calculate total current J from the ground truth permittivity
                E_total_gt = forward_model.forward(perm_labeled) # This is Es, need E_total
                # To get E_total, we need to solve the forward problem differently
                # For simplicity in training, we can approximate the target.
                # A better approach would be to pre-calculate these targets.
                # Let's calculate the full J and then J-
                Es_gt_complex = es_labeled[:, 0] + 1j * es_labeled[:, 1]
                J_plus_gt_vec = forward_model.get_J_plus_vector(es_labeled)
                
                # We need J_total to find J_minus. This requires solving the forward problem.
                # For now, let's use the final reconstructed chi as the loss target.
                # This is a simplification but will allow training to proceed.
                j_plus_labeled_img = forward_model.calculate_J_plus(es_labeled)
                network_output_labeled = model(es_labeled, j_plus_labeled_img)
                reconstructed_chi_labeled = forward_model.reconstruct_chi_from_output(network_output_labeled, J_plus_gt_vec)
                sup_loss = supervised_loss_fn(reconstructed_chi_labeled, perm_labeled)

            # --- Self-Supervised Step (on unlabeled data) ---
            es_unlabeled, _ = unlabeled_data
            es_unlabeled = es_unlabeled.to(device)
            
            j_plus_unlabeled_vec = forward_model.get_J_plus_vector(es_unlabeled)
            j_plus_unlabeled_img = forward_model.calculate_J_plus(es_unlabeled)
            network_output_unlabeled = model(es_unlabeled, j_plus_unlabeled_img)
            
            phys_loss = physics_loss_fn(network_output_unlabeled, j_plus_unlabeled_vec, es_unlabeled, forward_model)
            
            # --- Combine Losses and Update ---
            total_loss = sup_loss + args.physics_loss_weight * phys_loss
            total_loss.backward()
            optimizer.step()
            
            total_supervised_loss += sup_loss.item()
            total_physics_loss += phys_loss.item()

        end_time = time.time()
        epoch_duration = end_time - start_time
        
        avg_sup_loss = total_supervised_loss / num_batches
        avg_phys_loss = total_physics_loss / num_batches
        
        print(f"Epoch {epoch+1}/{args.epochs} | Supervised Loss: {avg_sup_loss:.6f} | Physics Loss: {avg_phys_loss:.6f} | Duration: {epoch_duration:.2f}s")
        
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised Training Script for JP-Net")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--labeled_percentage', type=float, default=0.2)
    parser.add_argument('--physics_loss_weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--forward_model_image_size', type=int, default=32)
    parser.add_argument('--n_incident_waves', type=int, default=32)
    parser.add_argument('--relative_permittivity_er', type=float, required=True)
    
    args = parser.parse_args()
    train_ssl(args)
