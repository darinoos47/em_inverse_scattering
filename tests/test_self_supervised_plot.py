# test_self_supervised_plot.py

import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F # Required for potential F.interpolate, though not used for Es anymore

# Import your custom modules
from dataset.scattering_dataset import ScatteringDataset
# We will load the UNet from torch.hub, so no need to import custom UNet class

def main():
    # --- Configuration ---
    data_dir = "./generated_dataset"
    checkpoint_path = "./checkpoints/unet_self_supervised_latest.pt" # Path to the trained self-supervised model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples_to_save = 5 # Adjust this to change how many samples you want to save
    save_dir = "./results/self_supervised_reconstructions" # Directory to save images
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # --- Load Model ---
    # Load the same UNet architecture as used in train_self_supervised.py
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=2, # Es has real/imag channels
        out_channels=1, # Permittivity has 1 channel
        init_features=36, # This is for feature maps, not spatial size
        pretrained=False
    ).to(device)

    # Load the trained weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Successfully loaded self-supervised model from {checkpoint_path}")
    else:
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}.")
        print("Please ensure the self-supervised model has been trained and saved.")
        return

    model.eval() # Set model to evaluation mode for inference

    # --- Load Data (Validation Set) ---
    full_dataset = ScatteringDataset(data_dir)
    # Split to get a validation set, similar to how it's done in training
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Use DataLoader to fetch samples one by one without shuffling
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Saving Results ---
    saved_samples_count = 0
    with torch.no_grad(): # Disable gradient calculations for inference
        for i, (Es_gt, perm_gt) in enumerate(val_loader):
            if saved_samples_count >= num_samples_to_save:
                break # Stop after saving the desired number of samples
            
            # Move data to the correct device
            Es_gt = Es_gt.to(device) # Es_gt is now 32x32
            perm_gt = perm_gt.to(device) # perm_gt is now 32x32

            # Get U-Net prediction
            # Es_gt is already 32x32, no resizing needed before model input
            pred_permittivity_raw = model(Es_gt) # U-Net output is 32x32

            # Convert tensors to numpy for saving
            gt_img_np = perm_gt.squeeze().cpu().numpy()
            recon_img_np = pred_permittivity_raw.squeeze().cpu().numpy()

            # Create a figure for each sample to save GT and Recon side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(6, 3)) # Single row, two columns for GT and Recon
            
            # Plot Ground Truth
            im_gt = axes[0].imshow(gt_img_np, cmap='viridis')
            axes[0].set_title(f"GT Sample {i}")
            axes[0].axis('off')
            fig.colorbar(im_gt, ax=axes[0], fraction=0.046, pad=0.04)

            # Plot Reconstructed
            im_recon = axes[1].imshow(recon_img_np, cmap='viridis')
            axes[1].set_title(f"Recon Sample {i}")
            axes[1].axis('off')
            fig.colorbar(im_recon, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            
            # Save the figure
            save_path = os.path.join(save_dir, f"sample_{i:04d}_reconstruction.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close(fig) # Close the figure to free memory

            print(f"Saved sample {i} to {save_path}")
            saved_samples_count += 1
    
    print(f"\n✅ Saved {saved_samples_count} samples to {save_dir}. You can find them in your project's results folder.")


if __name__ == "__main__":
    main()