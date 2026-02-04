import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================
# 1) PUTANJE
# =========================
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2) DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 3) PARAMETRI
# =========================
image_size = 64
batch_size = 128
z_dim = 100
lr = 0.0002
beta1 = 0.5
epochs = 20

# =========================
# 4) TRANSFORMACIJE
# =========================
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =========================
# 6) MODEL: GENERATOR
# =========================
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# =========================
# 7) MODEL: DISKRIMINATOR
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

# =========================
# 8) INIT TEŽINA (DCGAN standard)
# =========================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# =========================
# 10) TRENING FUNKCIJA
# =========================
def train():
    # =========================
    # 5) DATASET
    # =========================
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("Broj slika:", len(dataset))

    G = Generator(z_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    for epoch in range(1, epochs + 1):
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{epochs}]")

        for real, _ in loop:
            real = real.to(device)
            batch = real.size(0)

            # ======================
            # Train Discriminator
            # ======================
            noise = torch.randn(batch, z_dim, 1, 1, device=device)
            fake = G(noise)

            D_real = D(real)
            D_fake = D(fake.detach())

            loss_D_real = criterion(D_real, torch.ones_like(D_real))
            loss_D_fake = criterion(D_fake, torch.zeros_like(D_fake))
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ======================
            # Train Generator
            # ======================
            D_fake_for_G = D(fake)
            loss_G = criterion(D_fake_for_G, torch.ones_like(D_fake_for_G))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        # SAVE IMAGE GRID
        with torch.no_grad():
            fake_images = G(fixed_noise).detach().cpu()
            grid = make_grid(fake_images, nrow=8, normalize=True)
            save_image(grid, f"{OUTPUT_DIR}/epoch_{epoch}.png")

        # SAVE MODELS
        torch.save(G.state_dict(), f"{OUTPUT_DIR}/generator.pth")
        torch.save(D.state_dict(), f"{OUTPUT_DIR}/discriminator.pth")

    print("✅ Training finished! Check outputs/ folder.")

# =========================
# 11) GLAVNI POKRETANJE
# =========================
if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    train()
