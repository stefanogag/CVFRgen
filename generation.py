import torch
import matplotlib.pyplot as plt
from objects import *
from utils import retrieve_model
from prepare_dataset import *

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# --- Dataset and model configuration --- #
dataset_name = 'celeba'   # 'mnist', 'fer2013', 'celeba'
filename = 'Celebmodel50rec.pth'
generation_method = 'time_sample'

modelinfo = torch.load(filename, weights_only=True)
fullmodel = retrieve_model(modelinfo, dataset_name, device)

seed = None
if generation_method == 'disentangle' or generation_method == 'interpolate':
    dl, _, _ = get_dataloaders(dataset_name, modelinfo['dim'], 1, modelinfo['selected_classes'], 0.01)

    for X, _ in dl:
        seed = X[0]
        break

# --- Image generation --- #
N = 5          # Grid size for displaying images
fullmodel.eval()
generated_imgs = fullmodel.generate(
    mode=generation_method,   # Sampling mode: empirical method is recommended
    alpha=[0.5, 0.4],
    n_samples=N**2,
    img_class=0,         # Class label for generation (0 if single class)
    seed = seed,
    dir = [7, 1],        
).cpu()

# Plot generated images
fig, ax = plt.subplots(N, N, figsize=(10, 10))
for i in range(N):
    for j in range(N):
        ax[i, j].imshow(generated_imgs[i * N + j].permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.tight_layout()

plt.savefig("cholesky1.svg", format = 'svg')
plt.show()