"""

"""
from torch import nn, optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        # layers
        self.encoder = nn.Sequential(nn.Linear(80, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, latent_dim),
                                     nn.ReLU())

        self.decoder = nn.Sequential(nn.Linear(latent_dim, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, 80),
                                     nn.Sigmoid())
        # apply the sigmoid activation function to compress the output to a range of (0, 1)

    def forward(self, x):
        encoded = self.encoder(x)  # encode the input
        decoded = self.decoder(x)  # decode the input

        return encoded, decoded


class VAE(AutoEncoder):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        # mean vector and sd vector
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

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
            z = torch.randn(num_samples, self.num_hidden).to(device)
            samples = self.decoder(z)

        return samples


# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def loss_function(recon_x, x, mu, logvar):
    # Compute the binary cross-entropy loss between the reconstructed output and the input data
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # Compute the Kullback-Leibler divergence between the learned latent variable distribution
    # and a standard Gaussian distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Combine the two losses by adding them together and return the result
    return BCE + KLD


class VAEArchitectures:
    def __init__(self, latent_dim):
        self.model = VAE(latent_dim)
        self.objectives = {
            'loss': 0.0,
            'out-of-dist': 0.0
        }

    def train(self, train_loader, latent_dim, learning_rate=1e-3, epochs=10):
        model = VAE(latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                # get a batch of training data and move it to the device
                data = data.to(device)
                # forward pass
                encoded, decoded, mu, log_var = model(data)

                # compute the loss and perform backpropagation
                loss = loss_function(data, encoded, mu, log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.size(0)

            epoch_loss = total_loss / len(train_loader.dataset)
            print(
                "Epoch {}/{}: loss={:.4f}".format(epoch + 1, epochs, epoch_loss)
            )

        return model
