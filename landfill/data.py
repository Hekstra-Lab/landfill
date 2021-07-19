import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SingleImDataset(Dataset):
    def __init__(self, image, mask, cuda):

        if isinstance(image, str):
            self.image = np.load(image)
        else: self.image = image

        if isinstance(mask, str):
            self.mask = np.load(mask)
        else: self.mask = mask

        all_coords = torch.tensor(np.indices(self.image.shape).reshape(2,-1).T/np.array(self.image.shape)).to(torch.float)
        all_pixels = torch.tensor(self.image.reshape(-1)).to(torch.float)
        
        self.train_mask = ~self.mask.ravel()
        self.coords = all_coords[self.train_mask]
        self.pixels = all_pixels[self.train_mask]

        if cuda:
            self.coords = self.coords.cuda()
            self.pixels = self.pixels.cuda()

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]

    def __len__(self):
        return self.train_mask.sum()

def SingleImDataLoader(image, mask, batch_size=128, cuda=False):
    ds = SingleImDataset(image, mask, cuda=cuda)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
