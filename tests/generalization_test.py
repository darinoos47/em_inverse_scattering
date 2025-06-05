import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F # For interpolation

# Import your custom modules
from models.unet import UNet
from models.forward_model import InverseScattering # Needed to generate Es for the circle
from dataset.mnist_forward import load_mnist_dataset # Needed for the new flipped MNIST test

def generate_circle_permittivity(image_size=40, radius=8, center_x=20, center_y=20, permittivity_value=1.0):
    """
    Generates a 2D permittivity pattern representing a solid circle.

    Args:
        image_size (int): The width and height of the square image grid.
        radius (float): The radius of the circle.
        center_x (float): The x-coordinate of the circle's center.
        center_y (float): The y-coordinate of the circle's center.
        permittivity_value (float): The permittivity value inside the circle (typically 1.0 for contrast).

    Returns:
        torch.Tensor: A tensor of shape [1, 1, image_size, image_size]
                      representing the permittivity pattern, with values
                      between 0 and permittivity_value.
    """
    # Create a grid of coordinates
    x = torch.arange(image_size, dtype=torch.float32)
    y = torch.arange(image_size, dtype=torch.float32)
    # Create a 2D meshgrid
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    # Calculate the distance of each point from the center
    distance = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # Create the permittivity pattern: 1 inside the circle, 0 outside
    # The forward model scales input values [0, 1] by 'chai', so 1.0 means full contrast.
    permittivity_pattern = torch.zeros((image_size, image_size), dtype=torch.float32)
    permittivity_pattern[distance <= radius] = permittivity_value

    # Add batch and channel dimensions: [1, 1, H, W]
    return permittivity_pattern.unsqueeze(0).unsqueeze(0)

def plot_prediction_and_ground_truth(pred, target, title_prefix=""):
    """
    Plots a single prediction against its ground truth.

    Args:
        pred (torch.Tensor): Model prediction, expected shape [1, 1, H, W].
        target (torch.Tensor): Ground truth, expected shape [1, 1, H, W].
        title_prefix (str): Prefix for the plot titles.
    """
    # Squeeze to remove batch and channel dimensions for plotting (e.g., [H, W])
    pred_img = pred[0].squeeze().detach().cpu().numpy()
    target_img = target[0].squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the predicted image
    im1 = axes[0].imshow(pred_img, cmap='viridis')
    axes[0].set_title(f"{title_prefix} Predicted Permittivity")
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot the ground truth image
    im2 = axes[1].imshow(target_img, cmap='viridis')
    axes[1].set_title(f"{title_prefix} Ground Truth Permittivity")
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load a trained U-Net model and test its generalization
    on a synthetic 2D circle permittivity pattern and a flipped MNIST digit.
    """
    # --- Configuration ---
    # Path to your saved model checkpoint from train_loop.py
    checkpoint_path = "../checkpoints/unet_latest.pt"
    
    # Parameters for the forward model and UNet input/output
    # These should match the parameters used during dataset generation and UNet training
    forward_model_image_size = 40 # Image size used by InverseScattering (from generate_dataset.py)
    unet_output_image_size = 36   # Target image size for UNet output (from scattering_dataset.py interpolation)
    n_incident_waves = 36         # Number of incident waves (from generate_dataset.py)
    relative_permittivity_er = 15 # Relative permittivity used in forward model (from generate_dataset.py)

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Forward Model ---
    # This model is used to simulate the scattered fields (Es) for our synthetic circle
    forward_model = InverseScattering(
        image_size=forward_model_image_size,
        n_inc_wave=n_incident_waves,
        er=relative_permittivity_er
    ).to(device)
    forward_model.eval() # Set to evaluation mode

    # --- Initialize and Load UNet Model ---
    print("\n--- Loading trained UNet model ---")
    model = UNet(in_channels=2, out_channels=1).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded trained UNet from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}.")
        print("Please ensure the model has been trained and the checkpoint file exists.")
        return

    model.eval() # Set UNet to evaluation mode

    # --- Test 1: Synthetic 2D Circle Data ---
    print("\n--- Generating synthetic 2D circle data ---")
    # Define parameters for the circle
    circle_radius = 8
    circle_center_x = forward_model_image_size / 2
    circle_center_y = forward_model_image_size / 2

    # Generate the ground truth permittivity for the circle (at forward model's resolution)
    permittivity_circle_gt_40x40 = generate_circle_permittivity(
        image_size=forward_model_image_size,
        radius=circle_radius,
        center_x=circle_center_x,
        center_y=circle_center_y,
        permittivity_value=1.0 # Input to forward model should be in [0, 1]
    ).to(device)

    # Simulate scattered fields (Es) from the synthetic circle using the forward model
    with torch.no_grad():
        Es_circle = forward_model(permittivity_circle_gt_40x40) # Es_circle shape: [1, n_meas, n_inc]

    # Prepare Es for UNet input: stack real and imaginary parts
    # UNet expects input shape [B, 2, H, W] where H, W are n_meas, n_inc
    Es_unet_input_circle = torch.stack([Es_circle.real, Es_circle.imag], dim=1) # Shape: [1, 2, n_meas, n_inc]
    print(f"Simulated Es for circle. Shape for UNet input: {Es_unet_input_circle.shape}")

    # --- Perform Inference with UNet on Circle Es ---
    print("\n--- Performing UNet inference on circle data ---")
    with torch.no_grad():
        pred_circle = model(Es_unet_input_circle) # pred_circle shape: [1, 1, unet_output_image_size, unet_output_image_size]
    print(f"UNet prediction shape for circle: {pred_circle.shape}")

    # --- Plot Results for Circle ---
    print("\n--- Plotting results for 2D Circle Test ---")
    # Interpolate the original 40x40 circle ground truth to 36x36
    # to match the UNet's output resolution for fair comparison in plot.
    permittivity_circle_gt_36x36 = F.interpolate(
        permittivity_circle_gt_40x40,
        size=(unet_output_image_size, unet_output_image_size),
        mode='bilinear',
        align_corners=False
    )
    plot_prediction_and_ground_truth(pred_circle, permittivity_circle_gt_36x36, title_prefix="2D Circle Test")


    # --- Test 2: Flipped MNIST Digit ---
    print("\n--- Testing with a flipped MNIST digit ---")
    # Load MNIST dataset
    mnist_loader = load_mnist_dataset(image_size=forward_model_image_size, batch_size=1)

    # Get one sample from MNIST
    for i, (mnist_image, _) in enumerate(mnist_loader):
        if i == 0:
            # mnist_image is [1, 1, H, W]
            mnist_image_original = mnist_image.to(device) # Store original for unflipped test
            print(f"Original MNIST image shape: {mnist_image_original.shape}")

            # Flip the MNIST image upside down
            # Flipped along dimension 2 (height) which corresponds to up-down flip
            permittivity_flipped_mnist_gt_40x40 = torch.flip(mnist_image_original, dims=[2])
            print(f"Flipped MNIST image shape: {permittivity_flipped_mnist_gt_40x40.shape}")

            # Simulate scattered fields (Es) from the flipped MNIST image
            with torch.no_grad():
                Es_flipped_mnist = forward_model(permittivity_flipped_mnist_gt_40x40)
            
            # Prepare Es for UNet input: stack real and imaginary parts
            Es_unet_input_flipped_mnist = torch.stack([Es_flipped_mnist.real, Es_flipped_mnist.imag], dim=1)
            print(f"Simulated Es for flipped MNIST. Shape for UNet input: {Es_unet_input_flipped_mnist.shape}")

            # Perform Inference with UNet on Flipped MNIST Es
            with torch.no_grad():
                pred_flipped_mnist = model(Es_unet_input_flipped_mnist)
            print(f"UNet prediction shape for flipped MNIST: {pred_flipped_mnist.shape}")

            # Interpolate the original 40x40 flipped MNIST ground truth to 36x36
            # to match the UNet's output resolution for fair comparison in plot.
            permittivity_flipped_mnist_gt_36x36 = F.interpolate(
                permittivity_flipped_mnist_gt_40x40,
                size=(unet_output_image_size, unet_output_image_size),
                mode='bilinear',
                align_corners=False
            )

            # Plot Results for Flipped MNIST
            print("\n--- Plotting results for Flipped MNIST Test ---")
            plot_prediction_and_ground_truth(pred_flipped_mnist, permittivity_flipped_mnist_gt_36x36, title_prefix="Flipped MNIST Test")
            
            # --- New Test: Unflipped MNIST Digit ---
            print("\n--- Testing with an unflipped MNIST digit ---")
            # The original mnist_image_original is already 40x40 and unflipped.
            # Simulate scattered fields (Es) from the unflipped MNIST image
            with torch.no_grad():
                Es_unflipped_mnist = forward_model(mnist_image_original)
            
            # Prepare Es for UNet input: stack real and imaginary parts
            Es_unet_input_unflipped_mnist = torch.stack([Es_unflipped_mnist.real, Es_unflipped_mnist.imag], dim=1)
            print(f"Simulated Es for unflipped MNIST. Shape for UNet input: {Es_unet_input_unflipped_mnist.shape}")

            # Perform Inference with UNet on Unflipped MNIST Es
            with torch.no_grad():
                pred_unflipped_mnist = model(Es_unet_input_unflipped_mnist)
            print(f"UNet prediction shape for unflipped MNIST: {pred_unflipped_mnist.shape}")

            # Interpolate the original 40x40 unflipped MNIST ground truth to 36x36
            # to match the UNet's output resolution for fair comparison in plot.
            permittivity_unflipped_mnist_gt_36x36 = F.interpolate(
                mnist_image_original,
                size=(unet_output_image_size, unet_output_image_size),
                mode='bilinear',
                align_corners=False
            )

            # Plot Results for Unflipped MNIST
            print("\n--- Plotting results for Unflipped MNIST Test ---")
            plot_prediction_and_ground_truth(pred_unflipped_mnist, permittivity_unflipped_mnist_gt_36x36, title_prefix="Unflipped MNIST Test")

            break # Only process one MNIST sample (original, then flipped, then unflipped)

    print("\n✅ Generalization tests complete. Check the displayed plots.")

if __name__ == "__main__":
    main()


