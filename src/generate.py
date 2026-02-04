import torch
from dcgan_train import Generator, z_dim, device
from torchvision.utils import save_image, make_grid

# =========================
# Učitaj istrenirani generator
# =========================
G = Generator(z_dim).to(device)
G.load_state_dict(torch.load("outputs/generator.pth"))
G.eval()

# =========================
# Generiši nove slike
# =========================
noise = torch.randn(64, z_dim, 1, 1, device=device)
with torch.no_grad():
    fake_images = G(noise).cpu()

# =========================
# Napravi grid i sačuvaj
# =========================
grid = make_grid(fake_images, nrow=8, normalize=True)
save_image(grid, "outputs/generated_faces.png")
print("✅ Nova generisana slika spremljena: outputs/generated_faces.png")
