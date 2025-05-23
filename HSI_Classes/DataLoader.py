import torch as t
from torch.utils.data import Dataset

class HSI_dataset(Dataset):

    def __init__(self, preprocessed_df, logger):
        self.preprocessed_df = t.tensor(preprocessed_df)
        self.logger = logger

    def __len__(self):
        self.logger.trace("Dataset __len__")
        return len(self.preprocessed_df)

    def __getitem__(self, idx):
        data = self.preprocessed_df[idx]
        self.logger.trace("Dataset __getitem__")
        return data
