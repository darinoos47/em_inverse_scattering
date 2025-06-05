import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split

# Import your custom modules
from dataset.scattering_dataset import ScatteringDataset
from models.unet import UNet
from models.forward_model import InverseScattering
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

# --- Helper Functions (reused from other scripts) ---
def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_ssim_torch(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return ssim(pred_np, target_np, data_range=1.0)

def compute_es_loss(es_simulated, es_gt_stacked, loss_fn):
    """
    Computes MSE loss for complex scattered fields.
    es_simulated: [B, n_meas, n_inc] complex tensor from forward model.
    es_gt_stacked: [B, 2, n_meas, n_inc] real/imag stacked tensor from dataset.
    """
    es_gt_real = es_gt_stacked[:, 0, :, :]
    es_gt_imag = es_gt_stacked[:, 1, :, :]
    loss_real = loss_fn(es_simulated.real, es_gt_real)
    loss_imag = loss_fn(es_simulated.imag, es_gt_imag)
    return (loss_real + loss_imag).item()

# --- Main Debugging Logic ---
def main():
    # --- Configuration ---
    data_dir = "./generated_dataset"
    supervised_checkpoint_path = "./checkpoints/unet_supervised_initial.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # These parameters must match your dataset generation settings
    forward_model_image_size = 40
    n_incident_waves = 36
    # We will test different er values for Check 1, but this is the assumed 'true' er for the forward model instance below
    assumed_true_er_for_forward_model = 1.2 

    print(f"Using device: {device}")

    # --- Load Data Sample ---
    full_dataset = ScatteringDataset(data_dir)
    # Using a DataLoader with batch_size=1 and shuffle=False to get a consistent first sample
    val_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    # Get the first sample
    Es_gt, perm_gt = next(iter(val_loader))
    Es_gt = Es_gt.to(device) # Es_gt is [1, 2, n_meas, n_inc] (real, imag stacked)
    perm_gt = perm_gt.to(device) # perm_gt is [1, 1, H, W]

    print(f"\n--- Loaded Sample Details ---")
    print(f"Es_gt shape: {Es_gt.shape}, dtype: {Es_gt.dtype}, min/max: {Es_gt.min():.4f} / {Es_gt.max():.4f}")
    print(f"perm_gt shape: {perm_gt.shape}, dtype: {perm_gt.dtype}, min/max: {perm_gt.min():.4f} / {perm_gt.max():.4f}")

    # --- Initialize Models ---
    # The forward model will be used with different er values for testing, but
    # its parameters (image_size, n_inc_wave) remain consistent.
    # The 'main' forward_model instance below will use assumed_true_er_for_forward_model
    main_forward_model = InverseScattering(
        image_size=forward_model_image_size,
        n_inc_wave=n_incident_waves,
        er=assumed_true_er_for_forward_model # Use the er you believe dataset was generated with
    ).to(device)
    main_forward_model.eval()

    unet_model = UNet(in_channels=2, out_channels=1).to(device)
    if not os.path.exists(supervised_checkpoint_path):
        print(f"❌ Error: Supervised checkpoint not found at {supervised_checkpoint_path}")
        return
    checkpoint = torch.load(supervised_checkpoint_path, map_location=device)
    unet_model.load_state_dict(checkpoint['model_state_dict'])
    unet_model.eval()
    print(f"✅ Successfully loaded supervised UNet model from {supervised_checkpoint_path}")

    loss_fn_es = torch.nn.MSELoss()

    print("\n--- Debugging Checks ---")

    # --- Check 1: Verify 'er' for Data Generation ---
    # Pass the permittivity (perm_gt) through the forward solver with different 'er' values
    # and compare the resulting Es with Es_gt. The 'er' that yields the lowest loss
    # is likely the one used to generate the dataset.
    print("\n--- Check 1: Verifying Data Generation 'er' ---")
    test_er_values = [1.01, 1.1, 1.2, 1.5, 2.0, 5.0, 10.0, 15.0] # Common/test er values
    
    # The permittivity from the dataset (perm_gt) needs to be interpolated for the forward model.
    # The dataset perm_gt is 36x36, forward model input is 40x40.
    perm_gt_for_forward = F.interpolate(
        perm_gt,
        size=(forward_model_image_size, forward_model_image_size),
        mode='bilinear',
        align_corners=False
    ).clamp(0, 1) # Ensure values are in [0,1]

    with torch.no_grad():
        for test_er_value in test_er_values:
            # Create a temporary forward model instance for each test_er_value
            temp_forward_model = InverseScattering(
                image_size=forward_model_image_size,
                n_inc_wave=n_incident_waves,
                er=test_er_value
            ).to(device)
            temp_forward_model.eval()

            Es_recomputed = temp_forward_model(perm_gt_for_forward)
            es_loss = compute_es_loss(Es_recomputed, Es_gt, loss_fn_es)
            print(f"  Es Loss for perm_gt using er={test_er_value:.1f}: {es_loss:.6f}")
    print("  (Lower Es loss indicates a more likely 'er' for data generation)")


    # --- Check 2: Verify Supervised Model Permittivity Reconstruction ---
    # Pass Es_gt through the trained UNet and compare its permittivity output
    # with perm_gt (ground truth permittivity).
    print("\n--- Check 2: Supervised UNet Permittivity Reconstruction ---")
    with torch.no_grad():
        perm_pred_unet = unet_model(Es_gt) # UNet predicts permittivity [1, 1, 36, 36]
        
        psnr_val = compute_psnr(perm_pred_unet, perm_gt)
        ssim_val = compute_ssim_torch(perm_pred_unet, perm_gt)
        
        print(f"  UNet Output (perm_pred_unet) min/max: {perm_pred_unet.min():.4f} / {perm_pred_unet.max():.4f}")
        print(f"  PSNR (perm_pred_unet vs perm_gt): {psnr_val:.2f}")
        print(f"  SSIM (perm_pred_unet vs perm_gt): {ssim_val:.3f}")
    
    # For visualization reference
    perm_pred_unet_for_vis = perm_pred_unet.clone()


    # --- Check 3: Verify Consistency in Es Domain ---
    # Pass both UNet's predicted permittivity AND ground truth permittivity through the
    # main forward solver (configured with assumed_true_er_for_forward_model) and
    # compare their generated scattered fields.
    print("\n--- Check 3: Consistency between Permittivity and Es Domain ---")
    
    # 3a. Simulate Es from UNet's predicted permittivity
    perm_pred_unet_for_forward = F.interpolate(
        perm_pred_unet,
        size=(forward_model_image_size, forward_model_image_size),
        mode='bilinear',
        align_corners=False
    ).clamp(0, 1) # Clamp for forward model input
    
    with torch.no_grad():
        Es_from_pred_perm = main_forward_model(perm_pred_unet_for_forward)
        
        # 3b. Simulate Es from Ground Truth Permittivity
        Es_from_gt_perm = main_forward_model(perm_gt_for_forward) # perm_gt_for_forward from Check 1

        # Calculate loss between Es simulated from UNet's prediction and the original Es_gt
        loss_es_pred_vs_gt = compute_es_loss(Es_from_pred_perm, Es_gt, loss_fn_es)
        print(f"  Es Loss (from UNet's pred_perm vs original Es_gt): {loss_es_pred_vs_gt:.6f}")

        # Calculate loss between Es simulated from UNet's prediction and Es simulated from GT perm
        loss_es_pred_vs_gt_perm_simulated = compute_es_loss(Es_from_pred_perm, torch.stack([Es_from_gt_perm.real, Es_from_gt_perm.imag], dim=1), loss_fn_es)
        print(f"  Es Loss (from UNet's pred_perm vs from GT perm): {loss_es_pred_vs_gt_perm_simulated:.6f}")


    # --- Optional Visualization for Debugging ---
    # Plotting the GT Permittivity, UNet's Predicted Permittivity, and Clamped Permittivity
    print("\n--- Visualizing a Sample for Debug ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    gt_img = perm_gt.squeeze().detach().cpu().numpy()
    im0 = axes[0].imshow(gt_img, cmap='viridis')
    axes[0].set_title("Ground Truth Permittivity")
    fig.colorbar(im0, ax=axes[0])

    pred_unet_img = perm_pred_unet_for_vis.squeeze().detach().cpu().numpy()
    im1 = axes[1].imshow(pred_unet_img, cmap='viridis')
    axes[1].set_title(f"UNet Predicted Permittivity (Min: {pred_unet_img.min():.2f}, Max: {pred_unet_img.max():.2f})")
    fig.colorbar(im1, ax=axes[1])

    clamped_img = perm_pred_unet_for_forward.squeeze().detach().cpu().numpy()
    im2 = axes[2].imshow(clamped_img, cmap='viridis')
    axes[2].set_title(f"Clamped for Forward Model (Min: {clamped_img.min():.2f}, Max: {clamped_img.max():.2f})")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

    print("\n✅ Debugging script complete. Analyze the output and plots.")

if __name__ == "__main__":
    main()
