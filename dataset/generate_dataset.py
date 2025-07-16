# EISP/dataset/generate_dataset.py

from models.forward_model import InverseScattering
from dataset.mnist_forward import load_mnist_dataset, load_fashion_mnist_dataset
from dataset.generator_utils import generate_and_save_dataset
import time
import argparse

def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Generate dataset of permittivity -> scattered field pairs.")
    parser.add_argument('--image_size', type=int, default=32, help='Size of the image.')
    parser.add_argument('--n_inc', type=int, default=32, help='Number of incident waves.')
    parser.add_argument('--er', type=float, default=2.5, help='Relative permittivity.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for dataset shuffling')

    # Add an argument to choose the dataset
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'], help='Choose between "mnist" and "fashion_mnist".')
    
    # Add an argument for the output directory
    parser.add_argument('--output_dir', type=str, help='Output directory for the generated dataset. If not provided, a default will be used.')

    args = parser.parse_args()

    # Set a default output directory if one isn't provided
    if not args.output_dir:
        args.output_dir = f"/content/{args.dataset}_generated_dataset"
    
    print(f"Generating dataset using {args.dataset.upper()}...")

    scatter_model = InverseScattering(
        image_size=args.image_size,
        n_inc_wave=args.n_inc,
        er=args.er
    )

    # Load the selected dataset
    if args.dataset == 'mnist':
        dataloader = load_mnist_dataset(image_size=args.image_size, batch_size=64, seed=args.seed)
    else: # fashion_mnist
        dataloader = load_fashion_mnist_dataset(image_size=args.image_size, batch_size=64, seed=args.seed)


    # Generate and save the dataset
    generate_and_save_dataset(scatter_model, dataloader, save_dir=args.output_dir)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Dataset generation complete in {elapsed_time:.2f} seconds.")
    print(f"Dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
