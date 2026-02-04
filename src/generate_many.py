import os
import torch
from torchvision.utils import save_image
from dcgan_train import Generator, z_dim, device

OUTPUT_FOLDER = "fake_images"
NUM_IMAGES = 1000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

G = Generator(z_dim).to(device)
G.load_state_dict(torch.load("outputs/generator.pth", map_location=device))
G.eval()

with torch.no_grad():
    for i in range(NUM_IMAGES):
        noise = torch.randn(1, z_dim, 1, 1, device=device)
        fake = G(noise).cpu()

        # fake je [-1,1] → prebacimo u [0,1] za save
        fake = (fake + 1) / 2

        save_image(fake, f"{OUTPUT_FOLDER}/{i:05d}.png")

print(f"✅ Generisano {NUM_IMAGES} slika u folder: {OUTPUT_FOLDER}/")
