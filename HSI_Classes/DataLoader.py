import torch as t
from torch.utils.data import Dataset

class HSI_dataset(Dataset):
    def __init__(self, p_vec, logger):
        self.p_vec = t.tensor(p_vec)
        self.logger = logger

    def __len__(self):
        self.logger.trace("Dataset __len__")
        return len(self.p_vec)

    def __getitem__(self, idx):
        data = self.p_vec[idx]
        self.logger.trace("Dataset __getitem__")
        return data
