"""

"""
from torch import optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)


class AutoEncoder(nn.Module):
    def __init__(self, mid_layer, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.mid_layer = mid_layer

        # layers
        self.encoder = nn.Sequential(
            nn.Linear(88, mid_layer),
            nn.ReLU(),
            nn.Linear(mid_layer, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid_layer),
            nn.ReLU(),
            nn.Linear(mid_layer, 88),
            nn.Sigmoid(),
        )
        # apply the sigmoid activation function to compress the output to a range of (0, 1)

    def forward(self, x):
        encoded = self.encoder(x)  # encode the input
        decoded = self.decoder(encoded)  # decode the encoded output

        return encoded, decoded


class VAE(AutoEncoder):
    def __init__(self, mid_layer, latent_dim):
        super().__init__(mid_layer, latent_dim)
        # mean vector and sd vector
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.log_var = nn.Linear(self.latent_dim, self.latent_dim)

    def reparameterize(self, mu, log_var):
        sd = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sd)

        return mu + eps * sd

    def forward(self, x):
        encoded = self.encoder(x)

        # get the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)

        # reparameterize
        z = self.reparameterize(mu, log_var)

        decoded = self.decoder(z)
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            # generate random samples from distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)

        return samples


# # Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
# def loss_function(recon_x, x, mu, logvar):
#     # Compute the binary cross-entropy loss between the reconstructed output and the input data
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
#
#     # Compute the Kullback-Leibler divergence between the learned latent variable distribution
#     # and a standard Gaussian distribution
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     # Combine the two losses by adding them together and return the result
#     return BCE + KLD


class VAEArchitecture:
    def __init__(self, mid_layer, latent_dim):
        self.model = VAE(mid_layer, latent_dim)
        self.latent_dim = latent_dim
        self.mid_layer = mid_layer
        self.objectives = {
            'loss': 0.0,
            'OOD': 0.0
        }
        self.nondominated_rank = 0
        self.crowding_distance = 0.0

    def log_likelihood(self):
        ...

    def train(self, train_loader, learning_rate=1e-3, epochs=10):

        model = VAE(self.mid_layer, self.latent_dim).train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss(reduction="sum")

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, x in enumerate(train_loader):
                # get a batch of training data and move it to the device
                x = x.to(device)
                # forward pass
                encoded, decoded, mu, log_var = model(x)

                # compute the loss and perform backpropagation
                KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = criterion(decoded, x) + 3 * KLD

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)

            epoch_loss = total_loss / len(train_loader.dataset)
            print(
                "Epoch {}/{}: loss={:.4f}".format(epoch + 1, epochs, epoch_loss)
            )
            self.objectives['loss'] = epoch_loss

        return model

    def clone(self):
        return VAEArchitecture(self.latent_dim, self.mid_layer)
