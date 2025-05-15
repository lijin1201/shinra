import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import pandas as pd
import json

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
        image = tv_tensors.Image(torch.from_numpy(data).unsqueeze(0)) #.float())
        
        # Extract label from filename, e.g., "slice_062_3.npy" -> 3
        label = self.input.iloc[idx,1]
        # bbox = json.loads(self.input.iloc[idx].bb)
        mask = torch.zeros(image.shape)
        bbi = np.array(json.loads(self.input.iloc[idx].bb)).round() #.transpose(0,2,1)
        for b in bbi:
            # x1, y1, x2, y2 = b.round.int()
            mask[:,b[1,0]:b[1,1]+1,b[0,0]:b[0,1]+1] = 1.0
        mask = tv_tensors.Mask(mask)#,format="XYXY",canvas_size=data.shape)

        # Optional: apply transform (e.g., to tensor, normalize)
        if self.transform:
            image, mask = self.transform(image,mask)
        # else:
        #     data = torch.from_numpy(data).unsqueeze(0)  # Add channel dimension if needed
        
        return image, label, mask