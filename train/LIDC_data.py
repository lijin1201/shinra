import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class LIDC_Dataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.input = pd.read_csv(directory)
        
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        file_path = self.input.iloc[idx]
        data = np.load(file_path).astype(np.float32)  # or normalize here
        filename = os.path.basename(file_path)
        
        # Extract label from filename, e.g., "slice_062_3.npy" -> 3
        label_str = filename.split("_")[-1].split(".")[0]
        label = int(label_str)

        # Optional: apply transform (e.g., to tensor, normalize)
        if self.transform:
            data = self.transform(data)
        else:
            data = torch.from_numpy(data).unsqueeze(0)  # Add channel dimension if needed
        
        return data, label