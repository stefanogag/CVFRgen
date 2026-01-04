import torch
import torch.nn.functional as F
from torcheval.metrics import FrechetInceptionDistance as FID
from utils import retrieve_model
from prepare_dataset import *

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# --- Dataset and model configuration --- #
dataset_name = 'celeba'     # Options: 'mnist', 'fer2013', 'celeba'
filename = 'Celebmodel50rec.pth'      # Saved model file

# Load saved model and its metadata
modelinfo = torch.load(filename, weights_only=True)
selected_classes = modelinfo['selected_classes']
dim = modelinfo['dim']

# Rebuild the model and load its state
fullmodel = retrieve_model(modelinfo, dataset_name, device)
fullmodel.eval()

# Get test dataloader
_, _, dl = get_dataloaders(dataset_name, dim, batch_size=512, selected_classes=selected_classes)
print("Dataloader prepared.")

# --- Prepare real data --- #
real_datas, real_labels = [], []

for X, y in dl:
    real_datas.append(X)
    real_labels.append(y if fullmodel.CVFR.n_classes != 1 else torch.zeros_like(y))

real_imgs = torch.cat(real_datas, dim=0)
real_labels = torch.cat(real_labels, dim=0)

# --- Generate fake data matching real class distribution --- #
fake_images = []

for ea, label in enumerate(torch.unique(real_labels)):
    n_samples = (real_labels == label).sum()
    while len(fake_images * 1000) < n_samples:
        ea_gen = fullmodel.generate(
            mode='empirical',
            alpha=None,
            n_samples=1000,
            img_class=ea
        ).cpu()
        fake_images.append(ea_gen)
        print(f"Generated {len(fake_images * 1000)} / {n_samples} images for class {ea}")

fake_imgs = torch.cat(fake_images, dim=0)[: n_samples]
print(f"Generated {fake_imgs.shape[0]} fake images")

# --- Preprocess images for FID calculation --- #
real_imgs = F.interpolate(real_imgs.repeat(1, 3, 1, 1), size=(299, 299), mode='bilinear')
fake_imgs = F.interpolate(fake_imgs.repeat(1, 3, 1, 1), size=(299, 299), mode='bilinear')

# Build dataloaders for real and fake images
real_loader = DataLoader(real_imgs, batch_size=512)
fake_loader = DataLoader(fake_imgs, batch_size=512)

# --- Initialize and compute FID --- #
fid = FID(device=device)

# Update FID metric using batches to avoid memory overflow
print("Updaring FID data...")
with torch.no_grad():
    for batch_real, batch_fake in zip(real_loader, fake_loader):
        fid.update(batch_real.to(device), is_real=True)
        fid.update(batch_fake.to(device), is_real=False)

# Compute final FID score
print("Calculating FID score...")
fid_value = fid.compute()
print(f"FID Score: {fid_value.item():.4f}")
