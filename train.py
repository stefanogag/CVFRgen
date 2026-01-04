import torch
from utils import *
from objects import *
from prepare_dataset import *

# Select device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# --- Model initialization and hyperparameters --- #

dataset_name = 'celeba'    # Options: 'mnist', 'fer2013', 'celeba'
selected_classes = None     # Use None for all classes, or list e.g., [0]
filename = 'model.pth'      # Model save name

# Training configuration
batch_size = 256          
val_ratio = 0.1           
lr = 1e-3                 
epochs = 1000               

# Loss function parameters
weights = {
    'Rec': 50.,      # Reconstruction loss
    'Conv': 1.,       # Latent space convergence loss
    'Lyap': 0.,       # Lyapunov regularization
    'Mean': 1.,       # Centroid regularization
    'Perceptual': 0.1, # VGG perceptual loss
    'SSIM': 10.        # Structural Similarity loss
}
target_moments = None # Target covariance (if needed)

# Model architecture parameters
dim = 96          # Image dimension: 28 (MNIST), 48 (FER2013), 96 (CelebA)
encod_dim = 100    # Latent space dimension: 20 (MNIST), 50 (FER2013), 100 (CelebA)

CVFR_config = {
    'size': encod_dim,     # Size of latent representation
    'n_classes': 1,        # Number of classes
    'attractors': None,    # Optional fixed attractors
    'dt': 0.03,            # Time step for evolution
    'steps': 50,           # Number of integration steps
    'G': torch.eye(encod_dim), # Initial coupling matrix (if None, G is learned)
    'epsilon': 0.06        # Noise scale for SDE
}

hidden_dims = None #[600, 400, 200]  # Hidden layers' dimension for fully connected encoder/decoder

# --- Create Dataloaders --- #
train_dl, val_dl, test_dl = get_dataloaders(dataset_name, dim, batch_size, selected_classes, val_ratio)

# Retrieve shape of the dataset's images
for X, _ in train_dl:
    input_shape = X[0].shape
    break

# Build encoder and decoder
if dataset_name == 'mnist':
    encoder, decoder = build_autoencoder(input_shape, encod_dim, hidden_dims)
else:
    encoder = ConvEncoder(input_shape, encod_dim)
    decoder = ConvDecoder(encod_dim, input_shape)

# Initialize full model
fullmodel = FullModel(encoder, decoder, CVFR_config, device).to(device)

early_stopper = EarlyStopping(patience=20)
optimizer = torch.optim.Adam(fullmodel.parameters(), lr=lr)

if weights['Perceptual']:
    VGG = VGGPerceptualLoss(
        dim=dim,
        layers=['relu1_2', 'relu2_2', 'relu3_1', 'relu3_3', 'relu4_1']
    ).to(device)
else:
    VGG = None

# --- Training loop --- #
for epoch in range(epochs):
    print(f"----- Epoch {epoch+1} -----")

    FullTrain(train_dl, fullmodel, VGG, compute_losses, optimizer, weights)
    val_loss = FullTest(val_dl, fullmodel, VGG, compute_losses, weights, verbose=False)

    early_stopper(val_loss, fullmodel)
    if early_stopper.early_stop:
        break

# Restore the best model weights
early_stopper.restore_best_weights(fullmodel)

print("Final test on test set:")
FullTest(test_dl, fullmodel, VGG, compute_losses, weights)

# Compute covariance of latent space and store in model buffer
fullmodel.compute_covariance(train_dl)

# Save model and configuration
state = {
    'dim': dim,
    'encod_dim': encod_dim,
    'cvfr': CVFR_config,
    'hidden_dims': hidden_dims,
    'selected_classes': selected_classes,
    'state': fullmodel.state_dict()
}

torch.save(state, filename)
print(f"Saved PyTorch Model State and Info to {filename}")