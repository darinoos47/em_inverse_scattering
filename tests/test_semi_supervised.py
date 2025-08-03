# test/test_semi_supervised.py

import torch
import torch.nn as nn
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

# --- J-Net Architecture (Copied from training script for self-containment) ---
class JNet(nn.Module):
    def __init__(self, in_channels_es=2, in_channels_j=2, out_channels=1):
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

def compute_ssim_torch(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    data_range = target_np.max() - target_np.min()
    if data_range == 0: return 1.0
    return ssim(pred_np, target_np, data_range=data_range, channel_axis=None)

def generate_comparison_figure(gt_list, pred_list, output_dir, dataset_name, er_value):
    num_images = len(gt_list)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2.5, 5.5))
    vmin, vmax = 0.0, 1.0
    for i in range(num_images):
        gt_img_norm = gt_list[i].squeeze().detach().cpu().numpy()
        pred_img_norm = pred_list[i].squeeze().detach().cpu().numpy()
        axes[0, i].imshow(gt_img_norm, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        if i == 0: axes[0, i].set_ylabel('Ground Truth')
        im = axes[1, i].imshow(pred_img_norm, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0: axes[1, i].set_ylabel('Reconstructed')
    fig.suptitle(f'JP-Net Evaluation on {dataset_name} (Normalized Contrast)')
    fig.tight_layout(rect=[0, 0, 0.9, 0.93])
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    save_path = os.path.join(output_dir, f"{dataset_name}_JP-Net_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary figure saved to: {save_path}")

def evaluate_model(model, data_loader, device, output_dir, dataset_name, er_value, forward_model, num_eval_samples=5, seed=42):
    model.eval()
    total_loss, total_psnr, total_ssim = 0, 0, 0
    loss_fn = torch.nn.MSELoss()
    gt_images_to_plot, pred_images_to_plot = [], []

    print(f"\n--- Evaluating on {dataset_name} dataset ---")
    num_samples = len(data_loader.dataset)
    all_indices = list(range(num_samples))
    random.Random(seed).shuffle(all_indices)
    eval_indices = set(all_indices[:num_eval_samples])
    print(f"Will evaluate and visualize on {len(eval_indices)} samples with indices: {sorted(list(eval_indices))}")

    processed_samples = 0
    with torch.no_grad():
        for i, (Es_gt, perm_gt) in enumerate(data_loader):
            if i not in eval_indices:
                continue
            processed_samples += 1
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)

            # --- KEY CHANGE: Calculate J+ for the J-Net input ---
            j_plus_input = forward_model.calculate_J_plus(Es_gt)
            
            # --- KEY CHANGE: Pass both inputs to the model ---
            pred_permittivity = model(Es_gt, j_plus_input)

            total_loss += loss_fn(pred_permittivity, perm_gt).item()
            total_psnr += compute_psnr(pred_permittivity, perm_gt)
            total_ssim += compute_ssim_torch(pred_permittivity, perm_gt)
            gt_images_to_plot.append(perm_gt)
            pred_images_to_plot.append(pred_permittivity)

    if processed_samples > 0:
        avg_loss = total_loss / processed_samples
        avg_psnr = total_psnr / processed_samples
        avg_ssim = total_ssim / processed_samples
        print(f"Results for {dataset_name} (on {processed_samples} samples):")
        print(f"  Average Loss (MSE): {avg_loss:.6f}")
        print(f"  Average PSNR:       {avg_psnr:.2f} dB")
        print(f"  Average SSIM:       {avg_ssim:.4f}")
        generate_comparison_figure(gt_images_to_plot, pred_images_to_plot, output_dir, dataset_name, er_value)
    else:
        print("Warning: No samples were evaluated.")

def main():
    parser = argparse.ArgumentParser(description="Test a trained Semi-Supervised J-Net model.")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--mnist_data_dir', type=str, required=True)
    parser.add_argument('--fashion_mnist_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./test_results_ssl")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_eval_samples', type=int, default=5)
    parser.add_argument('--forward_model_image_size', type=int, default=32)
    parser.add_argument('--n_incident_waves', type=int, default=32)
    parser.add_argument('--relative_permittivity_er', type=float, required=True)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Testing J-Net from checkpoint: {args.checkpoint_path}")

    # Load the J-Net model architecture
    model = JNet().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded successfully.")

    # The forward model is always needed to calculate J+
    forward_model = InverseScattering(
        image_size=args.forward_model_image_size,
        n_inc_wave=args.n_incident_waves,
        er=args.relative_permittivity_er
    ).to(device)

    mnist_seed = args.seed
    fashion_mnist_seed = args.seed + 1

    mnist_dataset = ScatteringDataset(data_dir=args.mnist_data_dir)
    mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)
    evaluate_model(model, mnist_loader, device, args.output_dir, "MNIST", args.relative_permittivity_er, forward_model, num_eval_samples=args.num_eval_samples, seed=mnist_seed)

    fashion_dataset = ScatteringDataset(data_dir=args.fashion_mnist_data_dir)
    fashion_loader = DataLoader(fashion_dataset, batch_size=1, shuffle=False)
    evaluate_model(model, fashion_loader, device, args.output_dir, "Fashion-MNIST", args.relative_permittivity_er, forward_model, num_eval_samples=args.num_eval_samples, seed=fashion_mnist_seed)

if __name__ == "__main__":
    main()
