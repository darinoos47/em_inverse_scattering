# test/test_model.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import argparse
import sys
import random

# Add the project root to the Python path to allow for absolute imports
# This assumes the script is run from the project root directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.scattering_dataset import ScatteringDataset
from models.forward_model import InverseScattering

# --- Seeding Function for Reproducibility ---
def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Helper Functions (copied from train script for consistency) ---
def compute_psnr(img1, img2, max_val=1.0):
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_ssim_torch(pred, target):
    """Computes the Structural Similarity Index (SSIM) between two images."""
    # Detach tensors from the computation graph, move to CPU, and convert to NumPy
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    # Ensure data_range is appropriate for the image values
    data_range = target_np.max() - target_np.min()
    if data_range == 0:
        return 1.0 # Or handle as appropriate if images are uniform
    return ssim(pred_np, target_np, data_range=data_range)

def visualize_and_save_predictions(pred, gt, output_dir, dataset_name, sample_idx):
    """
    Visualizes the predicted and ground truth permittivity and saves the figure.
    """
    pred_img = pred.squeeze().detach().cpu().numpy()
    gt_img = gt.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Predicted Image
    im1 = axes[0].imshow(pred_img, cmap='viridis')
    axes[0].set_title("Predicted Permittivity")
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Ground Truth Image
    im2 = axes[1].imshow(gt_img, cmap='viridis')
    axes[1].set_title("Ground Truth Permittivity")
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f'Comparison for {dataset_name} - Sample Index {sample_idx}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure to the specified directory
    save_path = os.path.join(output_dir, f"{dataset_name}_comparison_{sample_idx:04d}.png")
    plt.savefig(save_path)
    plt.close(fig) # Close the figure to free memory

def evaluate_model(model, data_loader, device, output_dir, dataset_name, input_type, forward_model=None, num_visuals=5):
    """
    Evaluates the model on a given dataset and saves results.
    """
    model.eval()
    if forward_model:
        forward_model.eval()

    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    loss_fn = torch.nn.MSELoss()

    print(f"\n--- Evaluating on {dataset_name} dataset ---")
    
    # --- Deterministic selection of samples for visualization ---
    num_samples = len(data_loader.dataset)
    all_indices = list(range(num_samples))
    random.shuffle(all_indices) # This shuffle is deterministic due to the seed
    vis_indices = set(all_indices[:num_visuals])
    print(f"Will save visualizations for sample indices: {sorted(list(vis_indices))}")
    # ---

    with torch.no_grad():
        for i, (Es_gt, perm_gt) in enumerate(data_loader):
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)

            # Prepare the input for the U-Net based on the specified input_type
            if input_type == "BP":
                if forward_model is None:
                    raise ValueError("Forward model must be provided for input_type 'BP'")
                unet_input = forward_model.BP(Es_gt)
            else: # 'Es'
                unet_input = Es_gt
            
            # Get model prediction
            pred_permittivity = model(unet_input)

            # Calculate metrics
            loss = loss_fn(pred_permittivity, perm_gt)
            total_loss += loss.item()
            total_psnr += compute_psnr(pred_permittivity, perm_gt)
            total_ssim += compute_ssim_torch(pred_permittivity, perm_gt)

            # Save visualization for the selected random samples
            if i in vis_indices:
                visualize_and_save_predictions(pred_permittivity, perm_gt, output_dir, dataset_name, i)

    # Calculate average metrics
    avg_loss = total_loss / len(data_loader)
    avg_psnr = total_psnr / len(data_loader)
    avg_ssim = total_ssim / len(data_loader)

    # Print summary
    print(f"Results for {dataset_name}:")
    print(f"  Average Loss (MSE): {avg_loss:.6f}")
    print(f"  Average PSNR:       {avg_psnr:.2f} dB")
    print(f"  Average SSIM:       {avg_ssim:.4f}")
    print(f"  Visualizations saved to '{output_dir}'")
    print("------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Test a trained U-Net model on MNIST and Fashion-MNIST datasets.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--mnist_data_dir', type=str, required=True, help='Directory of the generated MNIST dataset.')
    parser.add_argument('--fashion_mnist_data_dir', type=str, required=True, help='Directory of the generated Fashion-MNIST dataset.')
    parser.add_argument('--output_dir', type=str, default="./test_results", help='Directory to save output figures.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible visualization sampling.')
    
    # Arguments needed to correctly initialize the model architecture
    parser.add_argument('--input_type', type=str, default="Es", choices=["BP", "Es"], help='Input to UNet: BP (backpropagation image) or Es (scattered field matrix). Must match the trained model.')
    parser.add_argument('--forward_model_image_size', type=int, default=32, help='Image size used by the forward model (for BP input type).')
    parser.add_argument('--n_incident_waves', type=int, default=32, help='Number of incident waves (for BP input type).')
    parser.add_argument('--relative_permittivity_er', type=float, default=2.5, help='Relative permittivity (for BP input type).')

    args = parser.parse_args()
    
    # --- Setup ---
    set_seed(args.seed) # Set the seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Random seed set to: {args.seed}")
    print(f"Loading checkpoint from: {args.checkpoint_path}")

    # --- Load Model ---
    unet_in_channels = 1 if args.input_type == "BP" else 2
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=unet_in_channels,
        out_channels=1,
        init_features=args.forward_model_image_size,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model loaded successfully.")

    # --- Initialize Forward Model (only if needed for BP input) ---
    forward_model = None
    if args.input_type == "BP":
        print("Initializing forward model for BP input generation.")
        forward_model = InverseScattering(
            image_size=args.forward_model_image_size,
            n_inc_wave=args.n_incident_waves,
            er=args.relative_permittivity_er
        ).to(device)

    # --- Test on MNIST ---
    mnist_dataset = ScatteringDataset(data_dir=args.mnist_data_dir)
    mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)
    evaluate_model(model, mnist_loader, device, args.output_dir, "MNIST", args.input_type, forward_model)

    # --- Test on Fashion-MNIST ---
    fashion_dataset = ScatteringDataset(data_dir=args.fashion_mnist_data_dir)
    fashion_loader = DataLoader(fashion_dataset, batch_size=1, shuffle=False)
    evaluate_model(model, fashion_loader, device, args.output_dir, "Fashion-MNIST", args.input_type, forward_model)

if __name__ == "__main__":
    main()

