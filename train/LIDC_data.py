import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class LIDC_Dataset(Dataset):
    def __init__(self, root_dir, metapath, transform=None):
        self.root_dir = root_dir
        self.metapath = metapath
        self.transform = transform
        self.input = pd.read_csv(metapath)
        
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir,self.input.iloc[idx,0])
        data = np.load(file_path)# .astype(np.float32)  # or normalize here
        
        # Extract label from filename, e.g., "slice_062_3.npy" -> 3
        label = self.input.iloc[idx,1]
        # label = int(label_str) if label_str.isdigit() else label_str

        # Optional: apply transform (e.g., to tensor, normalize)
        if self.transform:
            data = self.transform(torch.from_numpy(data).unsqueeze(0))
        else:
            data = torch.from_numpy(data).unsqueeze(0)  # Add channel dimension if needed
        
        return data, label