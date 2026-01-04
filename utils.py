import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim
from tqdm import tqdm
from objects import *
import copy

mse = nn.MSELoss()

class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual loss using selected layers of a pretrained VGG19.
    """

    def __init__(self, dim, layers=['relu1_2', 'relu2_2', 'relu3_1'], weights=None):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG19 weights

        # Map layer names to VGG19 indices
        self.layer_mapping = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13,
            'relu3_3': 15, 'relu4_1': 18
        }

        self.layers = layers
        self.d = dim
        self.layer_ids = [self.layer_mapping[l] for l in layers]
        self.weights = weights if weights else [1.0] * len(layers)

    def forward(self, x, y):
        """
        Compute perceptual loss between generated image x and target image y.
        """
        x = self._preprocess(x)
        y = self._preprocess(y)

        loss = 0.0
        x_feat, y_feat = x, y

        # Iterate through VGG layers and collect losses at specified layers
        for i, layer in enumerate(self.vgg):
            x_feat = layer(x_feat)
            y_feat = layer(y_feat)

            if i in self.layer_ids:
                w = self.weights[self.layer_ids.index(i)]
                loss += w * nn.functional.mse_loss(x_feat, y_feat)
        return loss

    def _preprocess(self, img):
        """Resize and normalize image for VGG input."""
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)  # Convert grayscale to 3 channels

        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        return (img - mean) / std


class EarlyStopping:
    def __init__(self, patience=10, delta=1e-5, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss for {self.counter} epoch(s).")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

    def restore_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def compute_losses(model, perc_model, x0, gen, z, y, cov_target, weights):
    """
    Compute various losses weighted by given coefficients.
    Returns a tensor containing each loss component.
    Args:
        model: The CVFR model.
        perc_model: Perceptual loss model.
        x0: Original input images.
        gen: Generated/reconstructed images.
        z: Latent representations at time T.
        y: Class labels.
        cov_target: Target covariance for Lyapunov regularization.
        weights: List of weights for each loss component.
    Returns:
        Tensor of shape (num_losses,) containing each weighted loss component.
    """
    l = torch.zeros(len(weights), device=model.device)

    if weights[0]:
        l[0] = weights[0] * mse(x0, gen)  # Reconstruction loss

    if weights[1]:
        l[1] = weights[1] * mse(z, model.attr[:, y].T)  # Attractor convergence loss

    if weights[2]:
        l[2] = weights[2] * model.lyapunov_reg(cov_target)  # Lyapunov regularization

    if weights[3]:
        for ea in range(4):  # Class centroid loss
            if (y == ea).any():
                l[3] = l[3] + ((z[y == ea].mean(dim=0) - model.attr[:, ea]) ** 2).mean()
        l[3] = weights[3] * l[3] / model.n_classes

    if weights[4]:
        l[4] = weights[4] * perc_model(gen, x0)  # Perceptual loss

    if weights[5]:
        l[5] = weights[5] * (1.0 - ssim(gen, x0, data_range=1.0))  # SSIM loss

    return l


def FullTrain(dataloader, model, perc_model, loss_fn, optimizer, betas={}, target_moments=None, verbose=True):
    """
    Full training loop for one epoch.
    """
    dev = model.device
    model.train()
    loop = tqdm(dataloader)
    running_losses = 0.0

    for i, (X, y) in enumerate(loop, 1):
        X = X.to(dev)
        y = (y if model.CVFR.n_classes != 1 else torch.zeros_like(y)).to(dev)

        # Forward pass
        gen, pred = model(X)
        losses = loss_fn(model.CVFR, perc_model, X, gen, pred, y, target_moments, list(betas.values()))
        loss = losses.sum()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_losses += losses
        avg_losses = running_losses / i

        # Progress bar display
        if verbose:
            msg = ''
            for j, ea in enumerate(betas.items()):
                msg += (ea[0] + f': {avg_losses[j].item():>7f}, ' if ea[1] else '')
            msg += f"Tot: {(avg_losses.sum()):>7f}"
            loop.set_postfix_str(msg)


def FullTest(dataloader, model, perc_model, loss_fn, betas={}, target_moments=None, verbose=True):
    """
    Evaluate the model on the test set.
    Computes both loss and classification accuracy.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    dev = model.device

    model.eval()
    test_losses = torch.zeros(len(betas), device=dev)
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(dev)
            y = (y if model.CVFR.n_classes != 1 else torch.zeros_like(y)).to(dev)

            # Forward pass
            gen, pred = model(X)

            test_losses += loss_fn(model.CVFR, perc_model, X, gen, pred, y, target_moments, list(betas.values()))

            # Classification by nearest attractor
            correct += (((pred[:, :, None] - model.CVFR.attr[None, :, :]) ** 2).sum(1).argmin(1) == y).float().sum().item()

    # Average losses
    test_losses /= num_batches
    test_loss = test_losses.sum()
    accuracy = correct / size

    if verbose:
        msg = ''
        for j, ea in enumerate(betas.items()):
            msg += (ea[0] + f': {test_losses[j].item():>7f}, ' if ea[1] else '')
        print("Test: " + msg + f" Tot: {test_loss:>7f}")
        print(f"Accuracy: {(100 * accuracy):>0.1f}%")

    return test_loss


def retrieve_model(info, dataset_name, device):
    """
    Reconstruct the model given saved info and trained state dict.
    Args:
        info (dict): Dictionary containing model parameters and state dict.
                     This should at least include:
                        - 'dim': Input image dimension.
                        - 'encod_dim': Latent space dimension.
                        - 'cvfr': CVFR configuration dictionary.
                        - 'hidden_dims': List of hidden layer sizes for the encoder/decoder (if required).
                        - 'state': State dictionary of the trained model.
        dataset_name (str): Name of the dataset ('mnist', 'fer2013', 'celeba').
        device (str): Device to load the model onto.
    """
    dim = info['dim']
    encod_dim = info['encod_dim']
    CVFR_config = info['cvfr']
    hidden_dims = info['hidden_dims']
    input_shape = (1, dim, dim)

    # Build encoder/decoder depending on dataset
    if dataset_name == 'mnist':
        encoder, decoder = build_autoencoder(input_shape, encod_dim, hidden_dims)
    else:
        encoder = ConvEncoder(input_shape, encod_dim)
        decoder = ConvDecoder(encod_dim, input_shape)

    # Combine into full model and load weights
    model = FullModel(encoder, decoder, CVFR_config, device).to(device)
    model.load_state_dict(info['state'])
    return model
