import torch
import os

#def generate_and_save_dataset(scatter_model, dataloader, save_dir="./generated_dataset", max_batches=200):
def generate_and_save_dataset(scatter_model, dataloader, save_dir="/content/generated_dataset", max_batches=200):

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scatter_model.eval()
    scatter_model.to(device)  # ✅ works for both CPU and GPU

    sample_id = 0  # counter for saving each sample

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            images = images.to(device)  # ✅ move input to correct device
            Es = scatter_model(images)  # [B, n_meas, n_inc]

            for j in range(images.shape[0]):
                Es_real = Es[j].real.cpu()
                Es_imag = Es[j].imag.cpu()
                permittivity = images[j].cpu()

                torch.save({
                    'Es_real': Es_real,
                    'Es_imag': Es_imag,
                    'permittivity': permittivity
                }, os.path.join(save_dir, f"sample_{sample_id:04d}.pt"))
                sample_id += 1

            if i % 10 == 0:
                print(f"Saved batch {i} on {device}")