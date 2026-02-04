# src/generate_batch.py
import torch
from dcgan_train import Generator, z_dim, device
from torchvision.utils import save_image, make_grid
import os

os.makedirs("outputs", exist_ok=True)

# Učitaj model
G = Generator(z_dim).to(device)
G.load_state_dict(torch.load("outputs/generator.pth"))
G.eval()

# Lista seedova
seeds = range(100)  # 0-99
batch_size = 16

for i in range(0, len(seeds), batch_size):
    batch_seeds = seeds[i:i+batch_size]
    torch.manual_seed(0)  # opcionalno: reproducibilnost
    noise = torch.stack([torch.randn(z_dim, 1, 1, device=device) for _ in batch_seeds])
    with torch.no_grad():
        fake_images = G(noise).cpu()
    grid = make_grid(fake_images, nrow=4, normalize=True)
    save_image(grid, f"outputs/generated_batch_{i}.png")
    print(f"✅ Batch {i}-{i+len(batch_seeds)-1} spremljen")
