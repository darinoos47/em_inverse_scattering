import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split

# Import your custom modules
from models.unet import UNet
from dataset.scattering_dataset import ScatteringDataset

def plot_sample_prediction(pred, target, title_suffix=""):
    """
    Plots a single prediction against its ground truth.

    Args:
        pred (torch.Tensor): Model prediction, expected shape [1, 1, H, W].
                             The first dimension is batch size, second is channel,
                             third and fourth are height and width.
        target (torch.Tensor): Ground truth, expected shape [1, 1, H, W].
        title_suffix (str): Additional text for the plot title (e.g., "Train Set" or "Test/Validation Set").
    """
    # Squeeze to remove batch and channel dimensions for plotting (e.g., [H, W])
    pred_img = pred[0].squeeze().detach().cpu().numpy()
    target_img = target[0].squeeze().cpu().numpy()

    # Create a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the predicted image
    im1 = axes[0].imshow(pred_img, cmap='viridis')
    axes[0].set_title(f"Predicted Permittivity {title_suffix}")
    axes[0].axis('off') # Hide axes ticks for cleaner image
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04) # Add colorbar

    # Plot the ground truth image
    im2 = axes[1].imshow(target_img, cmap='viridis')
    axes[1].set_title(f"Ground Truth Permittivity {title_suffix}")
    axes[1].axis('off') # Hide axes ticks for cleaner image
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04) # Add colorbar

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show() # Display the plot

def main():
    """
    Main function to load a trained U-Net model, perform inference on
    training and validation data, and visualize the results.
    """
    # --- Configuration ---
    # Directory where your generated dataset .pt files are stored
    data_dir = "../generated_dataset"
    # Path to your saved model checkpoint from train_loop.py
    checkpoint_path = "../checkpoints/unet_latest.pt"
    
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Dataset ---
    # Initialize the ScatteringDataset
    full_dataset = ScatteringDataset(data_dir)

    # Split the dataset into training and validation sets.
    # This split should ideally match how it was done during training
    # in train_loop.py to ensure consistent data partitioning.
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for both training and validation sets.
    # We use a batch_size of 1 for inference to easily visualize individual samples.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Initialize and Load Model ---
    # Instantiate the UNet model. Ensure in_channels and out_channels match
    # the model architecture you trained.
    # For scattered fields (real and imaginary parts) mapping to permittivity,
    # in_channels is 2 and out_channels is 1.
    model = UNet(in_channels=2, out_channels=1).to(device)

    # Load the trained model weights from the checkpoint file
    if os.path.exists(checkpoint_path):
        # Load checkpoint, mapping to the current device (CPU or GPU)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Load the model's state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded trained model from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}.")
        print("Please ensure the model has been trained and the checkpoint file exists.")
        return # Exit if the model cannot be loaded

    # Set the model to evaluation mode. This disables dropout and batch normalization
    # updates, ensuring consistent predictions.
    model.eval()

    # --- Plotting a Training Data Point ---
    print("\n--- Plotting a training sample prediction ---")
    with torch.no_grad(): # Disable gradient calculations for inference
        # Iterate through the training loader to get the first sample
        for i, (Es_train, perm_train) in enumerate(train_loader):
            if i == 0: # We only need one sample for visualization
                # Move the input data to the correct device
                Es_train, perm_train = Es_train.to(device), perm_train.to(device)
                # Perform a forward pass to get the model's prediction
                pred_train = model(Es_train)
                # Plot the prediction against the ground truth
                plot_sample_prediction(pred_train, perm_train, title_suffix="(Train Set)")
                break # Exit loop after getting the first sample

    # --- Plotting a Test/Validation Data Point ---
    print("\n--- Plotting a test/validation sample prediction ---")
    with torch.no_grad(): # Disable gradient calculations for inference
        # Iterate through the validation loader to get the first sample
        for i, (Es_val, perm_val) in enumerate(val_loader):
            if i == 0: # We only need one sample for visualization
                # Move the input data to the correct device
                Es_val, perm_val = Es_val.to(device), perm_val.to(device)
                # Perform a forward pass to get the model's prediction
                pred_val = model(Es_val)
                # Plot the prediction against the ground truth
                plot_sample_prediction(pred_val, perm_val, title_suffix="(Test/Validation Set)")
                break # Exit loop after getting the first sample

if __name__ == "__main__":
    main()


