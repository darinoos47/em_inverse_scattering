## This script generates the dataset of premittivity -> scattered field pairs using MNIST and the forward scattering model.

# dataset/generate_dataset.py

from models.forward_model import InverseScattering
from dataset.mnist_forward import load_mnist_dataset
from dataset.generator_utils import generate_and_save_dataset
import time
import argparse # Import the argparse module

def main():
    start_time = time.perf_counter()

    # Create the parser
    parser = argparse.ArgumentParser(description="Generate dataset of permittivity -> scattered field pairs.")

    # Add arguments
    parser.add_argument('--image_size', type=int, default=32,
                        help='Size of the image (e.g., 32 for 32x32).')
    parser.add_argument('--n_inc', type=int, default=32,
                        help='Number of incident waves.')
    parser.add_argument('--er', type=float, default=1.2,
                        help='Relative permittivity.')

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    image_size = args.image_size
    n_inc = args.n_inc
    er = args.er

    # Print the values
    print(f"Image Size: {image_size}")
    print(f"Number of Incident Waves (n_inc): {n_inc}")
    print(f"Relative Permittivity (er): {er}")

    # Instantiate forward model
    scatter_model = InverseScattering(
        image_size=image_size,
        n_inc_wave=n_inc,
        er=er
    )

    # Load MNIST
    mnist_loader = load_mnist_dataset(image_size=image_size, batch_size=64)

    # Generate and save dataset
    generate_and_save_dataset(scatter_model, mnist_loader)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()