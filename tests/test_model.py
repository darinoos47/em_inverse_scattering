# test/test_model.py (Updated)

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

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.scattering_dataset import ScatteringDataset
from models.forward_model import InverseScattering
# --- Import all possible model architectures ---
from models.dncnn import DnCNN
from models.swinir import SwinIR

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

# --- Model Loader ---
def get_model(model_name, in_channels, out_channels, image_size):
    """
    Loads and returns the specified model architecture.
    """
    print(f"Loading model architecture: {model_name}")
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
            out_channels=out_channels
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

# --- Helper Functions (compute_psnr, etc. remain the same) ---
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

def visualize_and_save_predictions(pred, gt, output_dir, dataset_name, sample_idx):
    pred_img = pred.squeeze().detach().cpu().numpy()
    gt_img = gt.squeeze().detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(pred_img, cmap='viridis'); axes[0].set_title("Predicted"); axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(gt_img, cmap='viridis'); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1])
    save_path = os.path.join(output_dir, f"{dataset_name}_comparison_{sample_idx:04d}.png")
    plt.savefig(save_path)
    plt.close(fig)

def evaluate_model(model, data_loader, device, output_dir, dataset_name, input_type, forward_model=None, num_visuals=5, seed=42):
    model.eval()
    if forward_model:
        forward_model.eval()

    total_loss, total_psnr, total_ssim = 0, 0, 0
    loss_fn = torch.nn.MSELoss()

    print(f"\n--- Evaluating on {dataset_name} dataset ---")
    
    num_samples = len(data_loader.dataset)
    all_indices = list(range(num_samples))
    random.Random(seed).shuffle(all_indices) # Use a seeded random instance
    vis_indices = set(all_indices[:num_visuals])
    print(f"Will save visualizations for sample indices: {sorted(list(vis_indices))}")

    with torch.no_grad():
        for i, (Es_gt, perm_gt) in enumerate(data_loader):
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)

            if input_type == "BP":
                net_input = forward_model.BP(Es_gt)
            else:
                net_input = Es_gt
            
            pred_permittivity = model(net_input)

            total_loss += loss_fn(pred_permittivity, perm_gt).item()
            total_psnr += compute_psnr(pred_permittivity, perm_gt)
            total_ssim += compute_ssim_torch(pred_permittivity, perm_gt)

            if i in vis_indices:
                visualize_and_save_predictions(pred_permittivity, perm_gt, output_dir, dataset_name, i)

    avg_loss = total_loss / len(data_loader)
    avg_psnr = total_psnr / len(data_loader)
    avg_ssim = total_ssim / len(data_loader)

    print(f"Results for {dataset_name}:")
    print(f"  Average Loss (MSE): {avg_loss:.6f}")
    print(f"  Average PSNR:       {avg_psnr:.2f} dB")
    print(f"  Average SSIM:       {avg_ssim:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Test a trained model.")
    # --- Add model_name argument ---
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'dncnn', 'swinir'], help='Name of the model architecture to test.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--mnist_data_dir', type=str, required=True)
    parser.add_argument('--fashion_mnist_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./test_results")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_type', type=str, required=True, choices=["BP", "Es"])
    parser.add_argument('--forward_model_image_size', type=int, default=32)
    parser.add_argument('--n_incident_waves', type=int, default=32)
    parser.add_argument('--relative_permittivity_er', type=float, required=True)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Testing {args.model_name} from checkpoint: {args.checkpoint_path}")

    # --- Load Correct Model Architecture ---
    in_channels = 1 if args.input_type == "BP" else 2
    model = get_model(
        args.model_name,
        in_channels=in_channels,
        out_channels=1,
        image_size=args.forward_model_image_size
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded successfully.")

    forward_model = None
    if args.input_type == "BP":
        forward_model = InverseScattering(
            image_size=args.forward_model_image_size,
            n_inc_wave=args.n_incident_waves,
            er=args.relative_permittivity_er
        ).to(device)

    # --- Test on MNIST ---
    mnist_dataset = ScatteringDataset(data_dir=args.mnist_data_dir)
    mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)
    evaluate_model(model, mnist_loader, device, args.output_dir, "MNIST", args.input_type, forward_model, seed=args.seed)

    # --- Test on Fashion-MNIST ---
    fashion_dataset = ScatteringDataset(data_dir=args.fashion_mnist_data_dir)
    fashion_loader = DataLoader(fashion_dataset, batch_size=1, shuffle=False)
    evaluate_model(model, fashion_loader, device, args.output_dir, "Fashion-MNIST", args.input_type, forward_model, seed=args.seed)

if __name__ == "__main__":
    main()