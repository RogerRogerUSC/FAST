import torch
import numpy as np
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import Sampler
from sklearn.datasets import fetch_openml
from scipy.stats import dirichlet

class GammaDataSampler(Sampler):
    def __init__(self, data_source, num_samples, shape, scale):
        self.data_source = data_source
        self.num_samples = num_samples
        self.shape = shape
        self.scale = scale
        self.indices = self.generate_gamma_indices()

    def generate_gamma_indices(self):
        indices = []
        for _ in range(self.num_samples):
            index = int(torch.distributions.gamma.Gamma(self.shape, self.scale).sample())
            indices.append(index % len(self.data_source))
        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples
