# models/dncnn.py
# A standard implementation of DnCNN

import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_layers=17, features=64):
        """
        Initializes the DnCNN model.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_layers (int): Total number of convolutional layers.
            features (int): Number of feature maps in the intermediate layers.
        """
        super(DnCNN, self).__init__()
        
        # First layer: Conv + ReLU
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        
        # Intermediate layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            
        # Final layer: Conv
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network. It learns the residual (noise).
        The input image is added to the network's output to get the clean image.
        """
        # The network learns the residual, which is subtracted from the input.
        # In our case, we want to map input -> output directly, so we return the output.
        # For a denoising task, you would return x - self.dncnn(x)
        # For a general reconstruction task like ours, we treat it as a direct mapping.
        return self.dncnn(x)


