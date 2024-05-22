import pandas as pd
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from vae import *

# Download the files
import gzip
from urllib import request

# data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data',
#                                                               transform=torchvision.transforms.ToTensor(),
#                                                               download=True),
#                                    batch_size=128,
#                                    shuffle=True)
#
# url = "http://yann.lecun.com/exdb/mnist/"
# filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
#              't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
# data = []
# for filename in filenames:
#     print("Downloading", filename)
#     request.urlretrieve(url + filename, filename)
#     with gzip.open(filename, 'rb') as f:
#         if 'labels' in filename:
#             # Load the labels as a one-dimensional array of integers
#             data.append(np.frombuffer(f.read(), np.uint8, offset=8))
#         else:
#             # Load the images as a two-dimensional array of pixels
#             data.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28))
#
# # Split into training and testing sets
# X_train, y_train, X_test, y_test = data
#
# # Normalize the pixel values
# X_train = X_train.astype(np.float32) / 255.0
# X_test = X_test.astype(np.float32) / 255.0
#
# # Convert labels to integers
# y_train = y_train.astype(np.int64)
# y_test = y_test.astype(np.int64)
#
# v_auto_encoder = VAEArchitecture(20, 8)
# v_auto_encoder.train(X_train)

class CustomDataSet(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # convert to tensor
        sample = torch.tensor(sample.values, dtype=torch.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample


dataset = CustomDataSet('quat_data.csv')

data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

vae = VAEArchitecture(40, 4)
vae.train(data_loader)
