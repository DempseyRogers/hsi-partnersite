import torch as t
from torch.utils.data import Dataset

class HSI_dataset(Dataset):

    def __init__(self, preprocessed_np, logger):
        self.preprocessed_np = t.tensor(preprocessed_np)
        self.logger = logger

    def __len__(self):
        self.logger.trace("Dataset __len__")
        return len(self.preprocessed_np)

    def __getitem__(self, idx):
        data = self.preprocessed_np[idx]
        self.logger.trace("Dataset __getitem__")
        return data
