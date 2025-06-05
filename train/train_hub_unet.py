# train/train_hub_unet.py

import torch
from torch.utils.data import DataLoader, random_split
from dataset.scattering_dataset import ScatteringDataset
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# Load U-Net from torch.hub
model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=2, out_channels=1, init_features=36, pretrained=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your dataset
dataset = ScatteringDataset(data_dir="./generated_dataset")
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Loss and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 20
checkpoint_path = "./checkpoints/hub_unet_latest.pt"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for Es, perm in train_loader:
        Es, perm = Es.to(device), perm.to(device)
        pred = model(Es)
        loss = loss_fn(pred, perm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Es, perm in val_loader:
            Es, perm = Es.to(device), perm.to(device)
            pred = model(Es)
            val_loss += loss_fn(pred, perm).item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

