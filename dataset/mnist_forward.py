## This code load the MNIST digit dataset, resize each image to match model size, and return a PyTorch DataLoader that can iterate over batches of images.

## These are standard PyTorch and torchvision imports:
## datasets: lets you download and use datasets like MNIST
## transforms: defines preprocessing steps (e.g., resizing, normalization)
## DataLoader: allows batching and shuffling of data for training

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def load_mnist_dataset(image_size=32, batch_size=64, seed=0):
    ## transforms.Compose([...]): chains together preprocessing steps
    ## transforms.ToTensor(): converts a PIL image to a PyTorch tensor with shape [1, H, W], normalized to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
        transforms.Resize((image_size, image_size))  # upscale to match Gd size
    ])
    dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform) ## 

    # Set seed for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    ## Wraps the dataset into a PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=g) 
    ## Each call to the dataloader will return: images, labels = next(iter(dataloader))
    ## images: tensor of shape [B, 1, image_size, image_size], values in [0, 1]
    ## labels: tensor of digit labels [B] — not used in your physics application but available
    return dataloader
    
def load_fashion_mnist_dataset(image_size=32, batch_size=64, seed=0):
    """
    Loads the Fashion-MNIST dataset and returns a PyTorch DataLoader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size))
    ])
    # Load the Fashion-MNIST dataset
    dataset = datasets.FashionMNIST(root='./fashion_mnist_data', train=True, download=True, transform=transform)

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=g)
    return dataloader    
