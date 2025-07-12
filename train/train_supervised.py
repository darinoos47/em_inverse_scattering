# train/train_supervised_BP.py

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
# We need to import UNet even if it's from hub to specify init_features if needed, though for supervised,
# we directly target the image quality.
# from models.unet import UNet # Not explicitly needed if using torch.hub.load
from models.forward_model import InverseScattering # To use the BP method

# --- Helper Functions ---
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
    data_dir="./generated_dataset",
    epochs=10,
    batch_size=8,
    lr=1e-4, # Learning rate, can be adjusted automatically using learning rate schedulers
    checkpoint_path="/content/checkpoints/unet_supervised_BP_latest.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), # PyTorch models and tensors must be explicitly moved to a device (CPU or GPU) to do operations. Here we get the available device.
    forward_model_image_size=32, # Ensure consistency with data generation
    n_incident_waves=32,         # Ensure consistency with data generation
    relative_permittivity_er=2.5, # Ensure consistency with data generation
    input_type="BP"
):
    """
    Trains the U-Net model in a supervised manner, taking Backpropagation (BP)
    results as input to refine them towards ground truth permittivity.
    Args:
        data_dir (str): Directory containing the generated dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        checkpoint_path (str): Path to save/load model checkpoints.
        device (torch.device): Device to run training on (cuda or cpu).
        forward_model_image_size (int): Image size used by the forward model.
        n_incident_waves (int): Number of incident waves for the forward model.
        relative_permittivity_er (float): Relative permittivity for the forward model.
    """
    # Load dataset & split into train/val
    full_dataset = ScatteringDataset(data_dir)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    # random_split is used to randomly divide a dataset into non-overlapping subsets of given sizes.
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # The PyTorch DataLoader is a powerful utility to load data from a Dataset, handle batching, shuffling, and parallel workers automatically. It saves us from having manually slice our data during training.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffling helps to avoid the model learning spurious patterns that come just from the order of the data rather than its true content.
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Load U-Net from torch.hub with in_channels=1 (BP) or 2 (Es)---
    if input_type == "BP":
        unet_in_channels = 1
    else:
        # Assuming Es_gt has shape [batch, num_probes, H, W]
        # you may want to adjust this
        unet_in_channels = 2

    # UNet now takes 1-channel permittivity image (from BP) as input
    # torch.hub is a PyTorch feature that lets you download and load popular models directly from GitHub, ready to use.
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=unet_in_channels, # Input is BP output (1-channel permittivity image)
        out_channels=1, # Output is reconstructed permittivity (1-channel)
        init_features=forward_model_image_size, # Consistent with previous setups
        pretrained=False # do not load pre-trained weights
    ).to(device)

    # Initialize Forward Model (used for BP method)
    forward_model = InverseScattering(
        image_size=forward_model_image_size,
        n_inc_wave=n_incident_waves,
        er=relative_permittivity_er
    ).to(device)
    # The forward model is used for its BP method, which is a fixed operation.
    # In PyTorch, calling .eval() on a model switches it to evaluation mode. As some layers in PyTorch behave differently during training and evaluation. This is important as we don't want train the forward model itself.
    forward_model.eval() 
    # An optimizer in PyTorch is what actually updates the model’s weights during training, based on the computed gradients from backpropagation.
    # Here we used ADAM optimizer. We give the parameters of the model that we want to optimize and the learning rate. We have kept the other parameters to their default values.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Loss function is now directly on permittivity images (U-Net output vs Ground Truth)
    loss_fn = torch.nn.MSELoss() 

    # --- Checkpoint Loading Logic ---    
    if os.path.exists(checkpoint_path):
        # map_location=device: ensures that when the checkpoint is loaded, its tensors (the weights) are moved to the correct device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # The following line loads the saved weights from your checkpoint back into your model so you can continue training or run inference.
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Resuming supervised BP training from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting supervised BP training from scratch.")
    # --- END Checkpoint Loading Logic ---

    # --- Initial Evaluation (before training starts) ---
    model.eval()
    initial_val_psnr_perm, initial_val_ssim_perm = 0, 0
    initial_val_loss = 0 # This will be MSE on permittivity
    
    print("\n--- Initial Model Performance (before supervised BP training) ---")
    # torch.no_grad() disables gradient calculation. It is useful for inference, when you are sure that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    with torch.no_grad():
        # The enumerate() function in Python is a built-in function that adds a counter to an iterable and returns it as an enumerate object.
        for i, (Es_gt, perm_gt) in enumerate(val_loader):
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)
            
            # Calculate BP output as UNet input
            if input_type == "BP":
                unet_input = forward_model.BP(Es_gt)
            else:
                unet_input = Es_gt
            # print(f'unet input shape: {unet_input.shape}') # Debugging line to check input shape
            pred_permittivity = model(unet_input)
            
            
            initial_val_psnr_perm += compute_psnr(pred_permittivity, perm_gt)
            initial_val_ssim_perm += compute_ssim_torch(pred_permittivity, perm_gt)
            initial_val_loss += loss_fn(pred_permittivity, perm_gt).item()
    
    initial_val_psnr_perm /= len(val_loader)
    initial_val_ssim_perm /= len(val_loader)
    initial_val_loss /= len(val_loader)
    
    print(f"Initial Val Loss (Perm): {initial_val_loss:.6f}")
    print(f"Initial Val PSNR (Permittivity): {initial_val_psnr_perm:.2f}")
    print(f"Initial Val SSIM (Permittivity): {initial_val_ssim_perm:.3f}")
    print(f"----------------------------------------------------\n")


    # Training Loop
    for epoch in range(epochs):
        start_time = time.time() 

        model.train() # Switch model to training mode
        train_loss = 0

        for batch_idx, (Es_gt, perm_gt) in enumerate(train_loader):
            Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)        

            if input_type == "BP":
                unet_input = forward_model.BP(Es_gt)
            else:
                unet_input = Es_gt

            pred_permittivity = model(unet_input)            
            
            # Loss is MSE between predicted permittivity and ground truth permittivity
            loss = loss_fn(pred_permittivity, perm_gt)

            # Before you calculate new gradients, you reset (zero) all previous gradients. If you don’t do this, PyTorch will accumulate gradients by default (gradient accumulation)
            optimizer.zero_grad()
            # The following line runs backpropagation on the loss. It computes gradients of the loss w.r.t. each model parameter
            loss.backward()
            # The following line uses those gradients to update the model weights, according to the optimizer’s rule (Adam here)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ----- Validation -----
        model.eval()
        val_loss, val_psnr_perm, val_ssim_perm = 0, 0, 0
        with torch.no_grad():
            for Es_gt, perm_gt in val_loader:
                Es_gt, perm_gt = Es_gt.to(device), perm_gt.to(device)
                
                if input_type == "BP":
                    unet_input = forward_model.BP(Es_gt)
                else:
                    unet_input = Es_gt
                # Calculate BP output as UNet input
                # bp_input_for_unet = forward_model.BP(Es_gt)
                pred_permittivity = model(unet_input)
                
                val_loss += loss_fn(pred_permittivity, perm_gt).item()
                val_psnr_perm += compute_psnr(pred_permittivity, perm_gt)
                val_ssim_perm += compute_ssim_torch(pred_permittivity, perm_gt)

        val_loss /= len(val_loader)
        val_psnr_perm /= len(val_loader)
        val_ssim_perm /= len(val_loader)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Perm): {avg_train_loss:.6f} | Val Loss (Perm): {val_loss:.6f} | Val PSNR (Perm): {val_psnr_perm:.2f} | Val SSIM (Perm): {val_ssim_perm:.3f} | Duration: {epoch_duration:.2f}s")

        # Visualize predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            # For visualization, take a sample from val_loader and predict
            with torch.no_grad():
                sample_Es_gt, sample_perm_gt = next(iter(val_loader))
                sample_Es_gt = sample_Es_gt.to(device)
                sample_perm_gt = sample_perm_gt.to(device)
                
                if input_type == "BP":
                    unet_input = forward_model.BP(sample_Es_gt)
                else:
                    unet_input = sample_Es_gt
                    
                sample_pred_permittivity = model(unet_input)
                
                visualize_predictions(sample_pred_permittivity, sample_perm_gt, epoch)

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    print("✅ Supervised BP training complete")

if __name__ == "__main__":
    print("Version 1")
    parser = argparse.ArgumentParser(description="Supervised training of UNet with BP input.")
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.') # Increased default epochs
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--checkpoint_path', type=str, default="/content/checkpoints/unet_supervised_BP_latest.pt", help='Path to save/load model checkpoints.')
    parser.add_argument('--forward_model_image_size', type=int, default=32, help='Forward model image size')
    parser.add_argument('--n_incident_waves', type=int, default=32, help='Number of incident waves')
    parser.add_argument('--relative_permittivity_er', type=float, default=1.2, help='Relative Permittivity')
    parser.add_argument('--data_dir', type=str, default="/content/generated_dataset", help='Directory containing the generated dataset (default: /content/generated_dataset).')
    parser.add_argument('--input_type', type=str, default="BP", choices=["BP", "Es"], help='Input to UNet: BP (backpropagation image) or Es (scattered field matrix)')

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint_path,
        forward_model_image_size=args.forward_model_image_size,
        n_incident_waves=args.n_incident_waves,
        relative_permittivity_er=args.relative_permittivity_er,
        data_dir=args.data_dir,
        input_type=args.input_type
    )
