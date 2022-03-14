import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torch.nn.functional as F

class TFCCDataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
    
    def __getitem__(self,index):
        img_name, label = self.data[index]
        arr_path = os.path.join(self.path, str(img_name)+'.npy')
        arr = np.load(arr_path)
        arr = np.reshape(arr, (256, 256, -1))
        arr = np.uint8(arr / arr.max() * 255)
        image = torch.FloatTensor(arr)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.data)
