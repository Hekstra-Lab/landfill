import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SingleImDataset(Dataset):
    def __init__(self, image, mask):
        all_coords = torch.tensor(np.indices(image.shape).reshape(2,-1).T/np.array(image.shape)).to(torch.float)
        all_pixels = torch.tensor(image.reshape(-1)).to(torch.float)
        
        self.train_mask = ~mask.ravel()
        self.coords = all_coords[self.train_mask]
        self.pixels = all_pixels[self.train_mask]

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]

    def __len__(self):
        return self.train_mask.sum()

