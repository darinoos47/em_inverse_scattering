# train/train_self_supervised.py

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import argparse
import time

# Import your custom modules
from dataset.scattering_dataset import ScatteringDataset
from models.forward_model import InverseScattering

# --- Helper Function for Total Variation Loss ---
def total_variation_loss(img):
    """
    Calculates the Anisotropic Total Variation loss for a batch of images.
    It sums the absolute differences between adjacent pixels horizontally and vertically.
    Args:
        img (torch.Tensor): Image tensor of shape [B, C, H, W].
    Returns:
        torch.Tensor: Scalar TV loss.
    """
    # Calculate horizontal differences
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    # Calculate vertical differences
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return (tv_h + tv_w)


def compute_psnr(img1, img2, max_val=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.
        max_val (float): Maximum possible pixel value of the image (e.g., 1.0 for normalized images).
    Returns:
        float: PSNR value.
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_ssim_torch(pred, target):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    Converts tensors to numpy arrays for skimage's SSIM function.
    Args:
        pred (torch.Tensor): Predicted image tensor.
        target (torch.Tensor): Ground truth image tensor.
    Returns:
        float: SSIM value.
    """
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return ssim(pred_np, target_np, data_range=1.0)

def visualize_predictions(pred_permittivity, gt_permittivity, epoch):
    """
    Visualizes the predicted permittivity and the ground truth permittivity.
    Args:
        pred_permittivity (torch.Tensor): Predicted permittivity from U-Net, shape [1, 1, H, W].
        gt_permittivity (torch.Tensor): Ground truth permittivity, shape [1, 1, H, W].
        epoch (int): Current epoch number for display in the title.
    """
    pred_img = pred_permittivity[0].squeeze().detach().cpu().numpy()
    target_img = gt_permittivity[0].squeeze().detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Predicted Permittivity")
    plt.imshow(pred_img, cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth Permittivity")
    plt.imshow(target_img, cmap='viridis')
    plt.colorbar()
    plt.suptitle(f"Epoch {epoch+1} - Permittivity Reconstruction")
    plt.tight_layout()
    plt.show()


def train(
    data_dir="/content/generated_dataset",
    epochs=10,
    batch_size=8,
    lr=1e-4,
    checkpoint_path="./checkpoints/unet_self_supervised_latest.pt", # Checkpoint for self-supervised training
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Parameters for the forward model (should match dataset generation)
    forward_model_image_size=32, # Changed to 32
    n_incident_waves=32,
    relative_permittivity_er=3,
    # Parameters for regularization
    l1_lambda=0.0, # Weight for L1 regularization on predicted permittivity
    tv_lambda=0.0, # Weight for Total Variation regularization on predicted permittivity
    weight_decay=0.0 # Weight for L2 regularization on network weights (optimizer parameter)
):
    """
    Trains the U-Net model using a self-supervised approach, including
    L1, Total Variation, and L2 (weight decay) regularization.
    Args:
        data_dir (str): Directory containing the generated dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        checkpoint_path (str): Path to save/load model checkpoints for self-supervised training.
        device (torch.device): Device to run training on (cuda or cpu).
        forward_model_image_size (int): Image size used by the forward model.
        n_incident_waves (int): Number of incident waves for the forward model.
        relative_permittivity_er (float): Relative permittivity for the forward model.
        l1_lambda (float): Weight for the L1 regularization term added to the loss.
                           Set to 0.0 to disable.
        tv_lambda (float): Weight for the Total Variation regularization term added to the loss.
                           Set to 0.0 to disable.
        weight_decay (float): Weight for L2 regularization (weight decay) applied to
                              the optimizer. Set to 0.0 to disable.
    """
    # Print all parameters for verification
    print("\n--- Training Parameters ---")
    print(f"Data Directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate (lr): {lr}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Forward Model Image Size: {forward_model_image_size}")
    print(f"Number of Incident Waves (n_incident_waves): {n_incident_waves}")
    print(f"Relative Permittivity (relative_permittivity_er): {relative_permittivity_er}")
    print(f"L1 Lambda: {l1_lambda}")
    print(f"TV Lambda: {tv_lambda}")
    print(f"Weight Decay: {weight_decay}")
    print("---------------------------\n")

    # Load dataset & split into train/val
    full_dataset = ScatteringDataset(data_dir)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Load U-Net from torch.hub ---
    # Now, Es_gt from ScatteringDataset and perm_gt from ScatteringDataset
    # should both have spatial dimensions of 32x32.
    # The torch.hub UNet will directly accept 32x32 input and produce 32x32 output.
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=2, # Es has real/imag channels
        out_channels=1, # Permittivity has 1 channel
        init_features=32, # This is for feature maps, not spatial size
        pretrained=False
    ).to(device)
    # --- END NEW ---

    # Initialize Forward Model (fixed, used for self-supervision loss)
    forward_model = InverseScattering(
        image_size=forward_model_image_size, # This is now 32
        n_inc_wave=n_incident_waves,
        er=relative_permittivity_er
    ).to(device)
    forward_model.eval()

    # Optimizer with weight_decay parameter for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn_es = torch.nn.MSELoss()

    # --- Checkpoint Loading Logic (Simplified) ---
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Resuming self-supervised training from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting self-supervised training from scratch (random initialization).")
    # --- END Checkpoint Loading Logic ---

    # --- Initial Evaluation (before training starts) ---
    model.eval() # Ensure model is in evaluation mode for initial metrics
    initial_val_psnr_perm, initial_val_ssim_perm = 0, 0
    initial_val_loss_es = 0

    print("\n--- Initial Model Performance (before self-supervised training) ---")
    print("--- DEBUG: Initial Evaluation Tensor States (first sample) ---")
    with torch.no_grad():
        for i, (Es_gt, perm_gt) in enumerate(val_loader):
            if i == 0: # Process only the first sample for detailed debug prints
                Es_gt_single = Es_gt[0:1].to(device) # Now 32x32
                perm_gt_single = perm_gt[0:1].to(device)

                print(f"Es_gt_single shape: {Es_gt_single.shape}, dtype: {Es_gt_single.dtype}")
                print(f"Es_gt_single min/max: {Es_gt_single.min():.4f} / {Es_gt_single.max():.4f}")
                print(f"perm_gt_single shape: {perm_gt_single.shape}, dtype: {perm_gt_single.dtype}")
                print(f"perm_gt_single min/max: {perm_gt_single.min():.4f} / {perm_gt_single.max():.4f}")

                # No resizing needed for UNet input, Es_gt is already 32x32
                pred_permittivity_raw_single = model(Es_gt_single) # UNet now receives 32x32 input
                print(f"pred_permittivity_raw_single shape: {pred_permittivity_raw_single.shape}, dtype: {pred_permittivity_raw_single.dtype}")
                print(f"pred_permittivity_raw_single min/max: {pred_permittivity_raw_single.min():.4f} / {pred_permittivity_raw_single.max():.4f}")

                # PSNR/SSIM directly compare UNet output (32x32) with perm_gt (32x32)
                single_psnr = compute_psnr(pred_permittivity_raw_single, perm_gt_single)
                single_ssim = compute_ssim_torch(pred_permittivity_raw_single, perm_gt_single)
                print(f"Single sample PSNR (Perm): {single_psnr:.2f}, SSIM (Perm): {single_ssim:.3f}")

                # No clamping/normalization for forward model input.
                # Just interpolation to match forward model image size (which is now 32x32)
                # So, effectively, no spatial change here if forward_model_image_size is 32.
                pred_permittivity_for_forward_single = F.interpolate(
                    pred_permittivity_raw_single, # This is 32x32 from UNet
                    size=(forward_model_image_size, forward_model_image_size), # Upscale/downscale to 32x32 for forward model
                    mode='bilinear',
                    align_corners=False
                )

                print(f"pred_permittivity_for_forward_single shape: {pred_permittivity_for_forward_single.shape}, dtype: {pred_permittivity_for_forward_single.dtype}")
                print(f"pred_permittivity_for_forward_single min/max: {pred_permittivity_for_forward_single.min():.4f} / {pred_permittivity_for_forward_single.max():.4f}")

                Es_simulated_single = forward_model(pred_permittivity_for_forward_single)
                print(f"Es_simulated_single shape: {Es_simulated_single.shape}, dtype: {Es_simulated_single.dtype}")
                print(f"Es_simulated_single real min/max: {Es_simulated_single.real.min():.4f} / {Es_simulated_single.real.max():.4f}")
                print(f"Es_simulated_single imag min/max: {Es_simulated_single.imag.min():.4f} / {Es_simulated_single.imag.max():.4f}")

                Es_gt_real_single = Es_gt_single[:, 0, :, :]
                Es_gt_imag_single = Es_gt_single[:, 1, :, :]
                loss_real_single = loss_fn_es(Es_simulated_single.real, Es_gt_real_single)
                loss_imag_single = loss_fn_es(Es_simulated_single.imag, Es_gt_imag_single)
                single_es_loss = (loss_real_single + loss_imag_single).item()
                print(f"Single sample Es Loss: {single_es_loss:.6f}")
                print("----------------------------------------------------\n")

            # Accumulate metrics for the full validation set
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)
            # No resizing needed for UNet input, Es_gt is already 32x32
            pred_permittivity_raw = model(Es_gt) # UNet now receives 32x32 input

            # PSNR/SSIM directly compare UNet output (32x32) with perm_gt (32x32)
            initial_val_psnr_perm += compute_psnr(pred_permittivity_raw, perm_gt)
            initial_val_ssim_perm += compute_ssim_torch(pred_permittivity_raw, perm_gt)

            # No clamping/normalization for forward model input.
            # Just interpolation to match forward model image size (which is now 32x32)
            # So, effectively, no spatial change here if forward_model_image_size is 32.
            pred_permittivity_for_forward = F.interpolate(
                pred_permittivity_raw, # This is 32x32 from UNet
                size=(forward_model_image_size, forward_model_image_size), # Upscale/downscale to 32x32 for forward model
                mode='bilinear',
                align_corners=False
            )
            Es_simulated = forward_model(pred_permittivity_for_forward)
            Es_gt_real = Es_gt[:, 0, :, :]
            Es_gt_imag = Es_gt[:, 1, :, :]
            loss_real = loss_fn_es(Es_simulated.real, Es_gt_real)
            loss_imag = loss_fn_es(Es_simulated.imag, Es_gt_imag)
            initial_val_loss_es += (loss_real + loss_imag).item()

    initial_val_psnr_perm /= len(val_loader)
    initial_val_ssim_perm /= len(val_loader)
    initial_val_loss_es /= len(val_loader)

    print(f"Initial Val Loss (Es): {initial_val_loss_es:.6f}")
    print(f"Initial Val PSNR (Permittivity): {initial_val_psnr_perm:.2f}")
    print(f"Initial Val SSIM (Permittivity): {initial_val_ssim_perm:.3f}")
    print(f"----------------------------------------------------\n")


    # Training Loop
    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss = 0

        for batch_idx, (Es_gt, perm_gt) in enumerate(train_loader):
            Es_gt = Es_gt.to(device)
            # No resizing needed for UNet input, Es_gt is already 32x32
            pred_permittivity_raw = model(Es_gt) # UNet now receives 32x32 input
            '''
            # For visualization, use the raw predicted permittivity
            if (epoch == 0 and batch_idx == 0) or (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    # Get one sample for visualization
                    vis_Es_gt = Es_gt[0:1].to(device)
                    vis_perm_gt = perm_gt[0:1].to(device)

                    # No resizing needed for UNet input for visualization, Es_gt is already 32x32
                    vis_pred_permittivity_raw = model(vis_Es_gt) # Get prediction for visualization

                    visualize_predictions(vis_pred_permittivity_raw, vis_perm_gt, epoch)
            '''

            # No clamping/normalization for forward model input.
            # Just interpolation to match forward model image size (which is now 32x32)
            # So, effectively, no spatial change here if forward_model_image_size is 32.
            pred_permittivity_for_forward = F.interpolate(
                pred_permittivity_raw, # This is 32x32 from UNet
                size=(forward_model_image_size, forward_model_image_size), # Upscale/downscale to 32x32 for forward model
                mode='bilinear',
                align_corners=False
            )

            Es_simulated = forward_model(pred_permittivity_for_forward)
            Es_gt_real = Es_gt[:, 0, :, :]
            Es_gt_imag = Es_gt[:, 1, :, :]

            loss_es_real = loss_fn_es(Es_simulated.real, Es_gt_real)
            loss_es_imag = loss_fn_es(Es_simulated.imag, Es_gt_imag)
            loss = loss_es_real + loss_es_imag # Base self-supervised Es loss

            # --- Add Regularization Terms (on network output) ---
            if l1_lambda > 0:
                l1_loss = torch.mean(torch.abs(pred_permittivity_raw)) # Mean L1 over predicted permittivity
                loss = loss + l1_lambda * l1_loss

            if tv_lambda > 0:
                tv_loss = total_variation_loss(pred_permittivity_raw) # Total Variation loss
                loss = loss + tv_lambda * tv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ----- Validation -----
        model.eval()
        val_loss_es, val_psnr_perm, val_ssim_perm = 0, 0, 0
        with torch.no_grad():
            for Es_gt, perm_gt in val_loader:
                Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)
                # No resizing needed for UNet input, Es_gt is already 32x32
                pred_permittivity_raw = model(Es_gt) # UNet now receives 32x32 input

                # PSNR/SSIM directly compare UNet output (32x32) with perm_gt (32x32)
                val_psnr_perm += compute_psnr(pred_permittivity_raw, perm_gt)
                val_ssim_perm += compute_ssim_torch(pred_permittivity_raw, perm_gt)

                # No clamping/normalization for forward model input.
                # Just interpolation to match forward model image size (which is now 32x32)
                # So, effectively, no spatial change here if forward_model_image_size is 32.
                pred_permittivity_for_forward = F.interpolate(
                    pred_permittivity_raw, # This is 32x32 from UNet
                    size=(forward_model_image_size, forward_model_image_size), # Upscale/downscale to 32x32 for forward model
                    mode='bilinear',
                    align_corners=False
                )
                Es_simulated = forward_model(pred_permittivity_for_forward)
                Es_gt_real = Es_gt[:, 0, :, :]
                Es_gt_imag = Es_gt[:, 1, :, :]
                val_loss_es_real = loss_fn_es(Es_simulated.real, Es_gt_real)
                val_loss_es_imag = loss_fn_es(Es_simulated.imag, Es_gt_imag)
                val_loss_es += (val_loss_es_real + val_loss_es_imag).item()

        val_loss_es /= len(val_loader)
        val_psnr_perm /= len(val_loader)
        val_ssim_perm /= len(val_loader)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Es): {avg_train_loss:.6f} | Val Loss (Es): {val_loss_es:.6f} | Val PSNR (Perm): {val_psnr_perm:.2f} | Val SSIM (Perm): {val_ssim_perm:.3f} | Duration: {epoch_duration:.2f}s")

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    print("✅ Self-supervised training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-supervised training for UNet inverse scattering model.")

    # Data and training parameters
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/content/generated_dataset",
        help='Directory containing the generated dataset (default: /content/generated_dataset).'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training (default: 8).'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for the optimizer (default: 1e-4).'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default="./checkpoints/unet_self_supervised_latest.pt",
        help='Path to save/load model checkpoints for self-supervised training (default: ./checkpoints/unet_self_supervised_latest.pt).'
    )
    # Note: 'device' is typically inferred or set via environment variables, not usually a direct argparse param
    # For simplicity, we'll keep it inferred in the train function.

    # Forward model parameters (should match dataset generation)
    parser.add_argument(
        '--forward_model_image_size',
        type=int,
        default=32,
        help='Image size used by the forward model (default: 32).'
    )
    parser.add_argument(
        '--n_incident_waves',
        type=int,
        default=32,
        help='Number of incident waves for the forward model (default: 32).'
    )
    parser.add_argument(
        '--relative_permittivity_er',
        type=float,
        default=3,
        help='Relative permittivity for the forward model (default: 1.2).'
    )

    # Regularization parameters (already existing)
    parser.add_argument(
        '--l1_lambda',
        type=float,
        default=0.0,
        help='Weight for L1 regularization on predicted permittivity (default: 0.0).'
    )
    parser.add_argument(
        '--tv_lambda',
        type=float,
        default=0.0,
        help='Weight for Total Variation regularization on predicted permittivity (default: 0.0).'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight for L2 regularization (weight decay) on network parameters (default: 0.0).'
    )

    args = parser.parse_args()

    # Determine device dynamically inside main
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint_path,
        device=device, # Pass the dynamically determined device
        forward_model_image_size=args.forward_model_image_size,
        n_incident_waves=args.n_incident_waves,
        relative_permittivity_er=args.relative_permittivity_er,
        l1_lambda=args.l1_lambda,
        tv_lambda=args.tv_lambda,
        weight_decay=args.weight_decay
    )