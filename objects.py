import torch
from torch import nn
from torch.distributions import MultivariateNormal
import itertools
import math
from scipy.linalg import solve_continuous_lyapunov
from contextlib import contextmanager


@contextmanager
def temporary_attrs(obj, **kwargs):
    old_values = {k: getattr(obj, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old_values.items():
            setattr(obj, k, v)


class CVFRLayer(nn.Module):
    """
    Continuous Variable Firing Rate (CVFR) Layer.

    This layer evolves input states under a stochastic differential equation
    towards class-specific attractors, optionally regularized by Lyapunov equation.

    Args:
        size (int): Dimensionality of the input features.
        n_classes (int): Number of attractors to be planted.
        attractors (torch.Tensor, optional): Matrix of attractor directions (size x n_classes).
        dt (float): Integration step size (Euler-Maruyama).
        steps (int): Number of integration steps.
        gamma (float): Nonlinearity sharpness parameter.
        beta (float): Feedback gain parameter.
        eig (float): Same eigenvalue for each planted attractorr.
        epsilon (float): Noise strength.
        G (torch.Tensor or nn.Parameter, optional): Noise covariance factor matrix.
        device (str): Device to run the computations on.
    """

    def __init__(self, size, n_classes, attractors=None, dt=0.03, steps=100,
                 gamma=1/8, beta=1., eig=1., epsilon=0.1, G=None, device='cpu'):
        super().__init__()
        self.device = device
        self.size = size
        self.n_classes = n_classes
        self.dt = dt
        self.steps = steps
        self.epsilon = epsilon
        self.eig = eig
        self.beta = beta
        self.gamma = gamma

        # Create projection matrix P and complementary projection antiP
        self.P = self._create_P(attractors)
        self.antiP = torch.eye(size, device=device) - self.P

        # Learnable system matrix A
        self.A = nn.Parameter(torch.empty((size, size), device=device))

        # Noise covariance factor matrix G (learnable if None provided)
        self.G = nn.Parameter(torch.empty((size, size), device=device)) if G is None else G.to(device)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        if self.G.requires_grad:
            nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, inputs, story=False):
        """
        Forward pass using Euler-Maruyama integration.
        Optionally stores the entire trajectory if story=True.
        """
        x = inputs
        B = (self.A @ self.antiP + self.eig * self.P).T

        if story:
            self.trajectory = torch.zeros(self.steps + 1, x.shape[0], self.size, device=self.device)
            self.trajectory[0] = x.detach()

        for i in range(self.steps):
            # Stochastic noise term
            gaussian_noise = torch.randn_like(x)
            noise = self.epsilon * gaussian_noise @ self.G.T

            # Compute deterministic and stochastic updates
            x_dot = -x + self.beta * torch.matmul(self.non_linearity(x), B)
            x = x + self.dt * x_dot + math.sqrt(self.dt) * noise

            if story:
                self.trajectory[i + 1] = x.detach()

        return x

    def _create_P(self, attr):
        """
        Construct projection matrix P onto the attractor subspace.
        Attractors are expected to be orthogonal with entries as described in the article.
        """
        if attr is None:
            # Attractors are created as default "one-hot" vectors
            attr = torch.zeros(self.size, self.n_classes, device=self.device)
            block = self.size // self.n_classes
            for c in range(self.n_classes):
                attr[c * block:(c + 1) * block, c] = 1

            # Only biggest element of the alphabet is used
            a = self.beta * self.eig / 2 + math.sqrt((self.beta * self.eig / 2) ** 2 - self.gamma)
            self.attr = attr * a

        elif torch.is_tensor(attr):
            # If attractors are provided as a tensor, use them directly
            attr = attr.to(self.device)
            self.attr = attr
            
        elif attr == 'alt':
            # Alternative construction where also some "spurious" attractors are used to classify
            n = math.ceil(math.log2(self.n_classes+1))
            attr = torch.zeros(self.size, n, device=self.device)
            block = self.size // n
            for c in range(n):
                attr[c * block:(c + 1) * block, c] = 1
            self.attr = attr
            for d in itertools.product([0, 1], repeat=n):
                d = torch.tensor(d, dtype=torch.float32, device=self.device)[:, None]
                if d.sum() <= 1:
                    continue
                self.attr = torch.cat([self.attr, attr@d], dim=1)
                if self.attr.shape[1] == self.n_classes:
                    break

            # Only biggest element of the alphabet is used
            a = self.beta * self.eig / 2 + math.sqrt((self.beta * self.eig / 2) ** 2 - self.gamma)
            self.attr *= a

        norm_attr = attr / (attr.norm(dim=0, keepdim=True) + 1e-8)**2
        return norm_attr @ attr.T

    def non_linearity(self, x):
        return x ** 2 / (self.gamma + x ** 2)

    def der_non_linearity(self, x):
        return 2 * self.gamma * x / (self.gamma + x ** 2) ** 2

    def lyapunov_reg(self, target_moments):
        """
        Lyapunov regularization to enforce local stability of dynamics.
        Encourages satisfaction of the Lyapunov equation near attractors.
        """
        B = self.A @ self.antiP + self.eig * self.P
        J = -torch.eye(self.size, device=self.device)[None] + \
            self.beta * self.der_non_linearity(self.attr.T)[:, None, :] * B[None]

        # Residual of Lyapunov equation: J M + M J^T + GG^T ≈ 0
        residual = J @ target_moments + target_moments @ J.mT + self.G @ self.G.T
        return (residual ** 2).mean()

    def extra_repr(self):
        return f"dim={self.size}, attractors={self.n_classes}"


class ConvEncoder(nn.Module):
    """Convolutional encoder for image to latent vector mapping."""

    def __init__(self, input_shape, encod_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        # Determine flattened dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            self.flattened_dim = dummy_output.view(1, -1).size(1)

        self.fc_to_latent = nn.Linear(self.flattened_dim, encod_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc_to_latent(x)
        return x


class ConvDecoder(nn.Module):
    """Convolutional decoder to reconstruct images from latent vectors."""

    def __init__(self, encod_dim, output_shape):
        super().__init__()
        upscale_factor = 2 ** 4
        self.H, self.W = output_shape[1] // upscale_factor, output_shape[2] // upscale_factor
        self.fc_from_latent = nn.Linear(encod_dim, 512 * self.H * self.W)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, output_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc_from_latent(z)
        x = x.view(-1, 512, self.H, self.W)
        x = self.decoder(x)
        return x


class Reshape(nn.Module):
    """Utility layer to reshape tensors."""

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


def build_autoencoder(img_shape, latent_dim, hidden_dims=[512, 256]):
    """
    Build a standard fully connected autoencoder (encoder + decoder).
    """
    C, H, W = img_shape
    flat_dim = C * H * W

    # ENCODER: Flatten -> hidden layers -> latent
    encoder_layers = [nn.Flatten()]
    dims = [flat_dim] + hidden_dims + [latent_dim]
    for i in range(len(dims) - 1):
        encoder_layers += [
            nn.Linear(dims[i], dims[i+1]),
            nn.ReLU()
        ]
    encoder = nn.Sequential(*encoder_layers)

    # DECODER: latent -> hidden layers reversed -> reshape
    decoder_layers = []
    rev_dims = dims[::-1]
    for i in range(len(rev_dims) - 2):
        decoder_layers += [
            nn.Linear(rev_dims[i], rev_dims[i+1]),
            nn.ReLU()
        ]
    decoder_layers += [
        nn.Linear(rev_dims[-2], rev_dims[-1]),
        nn.Sigmoid(),
        Reshape(C, H, W)]
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


class FullModel(nn.Module):
    """
    Full model: encoder -> CVFR evolution -> decoder.
    Supports generation sampling and Lyapunov-based covariance estimation.

    Args:
        encoder (nn.Module): Neural network that maps input images to latent vectors.
        decoder (nn.Module): Neural network that reconstructs images from latent vectors.
        CVFR_config (dict): Configuration dictionary for CVFRLayer.
        device (str): Device to run the model on.
    """

    def __init__(self, encoder, decoder, CVFR_config, device):
        super().__init__()
        self.device = device

        # Buffer to store class-specific covariance matrices
        self.register_buffer(
            'covs', torch.empty(CVFR_config['n_classes'], CVFR_config['size'], CVFR_config['size'])
        )

        self.Encod = encoder.to(device)
        self.CVFR = CVFRLayer(**CVFR_config, device=device)
        self.Decod = decoder.to(device)

    def forward(self, inputs):
        x_enc = self.Encod(inputs)
        x_CVFR = self.CVFR(x_enc)
        x_gen = self.Decod(x_CVFR)
        return x_gen, x_CVFR

    def compute_covariance(self, dataloader):
        """Estimate covariance for each class using encoded samples."""
        with torch.no_grad():
            for ea in range(self.CVFR.n_classes):
                XT = []
                for X, y in dataloader:
                    X = X.to(self.device)
                    y = (y if self.CVFR.n_classes != 1 else torch.zeros_like(y)).to(self.device)
                    XT.append(self.CVFR(self.Encod(X[y == ea])).cpu())
                pointsT = torch.cat(XT, dim=0)
                self.covs[ea] = torch.cov(pointsT.T)

    def generate(self, img_class, mode='empirical', alpha=None, n_samples=1, seed=None, dir=0):
        """Generate samples using different modes.
        img_class: class index for the attractor.
        mode: generation method.
            avilable modes:
                'time_sample': Langevin-like time sampling. 
                'lyapunov': sample from Gaussian with Lyapunov covariance (scaled by alpha if provided)
                'empirical': sample from Gaussian with empirical covariance
                'interpolate': linear interpolation between two seed images (requires seed with 2 images)
                'disentangle': traverse along two directions in latent space (requires alpha and seed img)
        """
        with torch.no_grad():
            if mode == 'time_sample':
                # The CVFR noise covariance is substituted with a G_eq such that the asymptotic 
                # covariance satisfying the Lyapunov equation is the empirical one
                G_eq = self.cholesky(img_class)
                x0 = self.CVFR.attr[:, img_class].unsqueeze(0)
                with temporary_attrs(self.CVFR, G=G_eq, steps=n_samples):
                    _ = self.CVFR(x0.to(self.device), story=True)
                in_sample = self.CVFR.trajectory.squeeze(1)

            elif mode == 'lyapunov':
                alpha = alpha if alpha is not None else self.CVFR.epsilon
                in_sample = self.sample_from_distribution(n_samples, img_class, alpha)

            elif mode == 'empirical':
                in_sample = self.sample_from_distribution(n_samples, img_class, alpha=None)

            elif mode == 'interpolate':
                assert torch.is_tensor(seed) and seed.shape[0]==2, 'seed must be a tensor of shape (2, C, H, W)'
                z = self.Encod(seed.to(self.device))
                zT = self.CVFR(z)
                ver = (zT[1] - zT[0]) / torch.linalg.vector_norm(zT[1] - zT[0])
                slide = torch.linspace(0, 1, steps=n_samples, device=self.device)[:, None] * ver[None, :]
                in_sample = zT[0] + slide

            elif mode == 'disentangle':
                assert alpha is not None, 'alpha must be provided for disentangle mode'
                alpha = alpha if isinstance(alpha, list) else [alpha, alpha]
                # Requires a seed image and two directions (int), corresponding to the principal
                # components of the empirical covariance. Here alpha is the step size along the direction dir.
                # Returns a grid of (n_samples x n_samples) images.
                x_enc = self.Encod(seed.unsqueeze(0).to(self.device))
                in_sample = self.CVFR(x_enc.repeat(n_samples, 1))

                eigvals, eigvecs = torch.linalg.eig(self.covs[0])
                sorted_indices = torch.argsort(eigvals.real, descending=True)
                eigvecs_sorted = eigvecs[:, sorted_indices].real
                v1, v2 = eigvecs_sorted[:, dir[0]], eigvecs_sorted[:, dir[1]]

                range = torch.arange(math.sqrt(n_samples), device=self.device)
                grid_slide = torch.cartesian_prod(range * alpha[0], range * alpha[1])
                slider = grid_slide[:, 0][:, None] * v1[None, :] + grid_slide[:, 1][:, None] * v2[None, :]
                in_sample = in_sample + slider

            return self.Decod(in_sample.to(self.device))

    def sample_from_distribution(self, n_samples, img_class, alpha):
        """
        Sample from a Gaussian distribution around an attractor.
        If alpha is provided, variance is scaled by alpha.
        """
        x0 = self.CVFR.attr[:, img_class]

        cov = self.covs[img_class].to(x0.device) if alpha is None else alpha ** 2 * self.lyapunov_cov(x0).to(x0.device)
        cov = (cov + cov.T) / 2  # Ensure numerical symmetry

        return MultivariateNormal(loc=x0, covariance_matrix=cov).sample(sample_shape=(n_samples,))

    def lyapunov_cov(self, x0):
        """Compute the solution Σ of the Lyapunov equation"""
        G = self.CVFR.G.detach()
        noise_cov = G @ G.T
        J = self.jacobian(x0)
        Sigma = solve_continuous_lyapunov(J.cpu(), -noise_cov.cpu())
        return torch.tensor(Sigma)

    def jacobian(self, x0):
        B = (self.CVFR.A @ self.CVFR.antiP + self.CVFR.eig * self.CVFR.P).detach()
        dF = self.CVFR.der_non_linearity(x0)
        J = -torch.eye(x0.shape[-1], device=self.device) + self.CVFR.beta * B @ torch.diag(dF)
        return J
    
    def cholesky(self, img_class):
        """Compute Cholesky factor from empirical covariance and Jacobian."""
        x0 = self.CVFR.attr[:, img_class]
        J = self.jacobian(x0)

        lhs = -(J @ self.covs[img_class] + self.covs[img_class].T @ J.T) / self.CVFR.epsilon ** 2
        lhs = 0.5 * (lhs + lhs.T)  # Symmetrize

        # Ensure positive semi-definiteness
        eigvals, eigvecs = torch.linalg.eigh(lhs)
        eigvals_clipped = torch.clamp(eigvals, min=1e-7)
        lhs_psd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

        return torch.linalg.cholesky(lhs_psd)
