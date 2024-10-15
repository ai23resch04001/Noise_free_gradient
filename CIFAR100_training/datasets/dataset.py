
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from .folder import ImageFolder

from .utils import check_integrity, download_and_extract_archive
from .vision import VisionDataset

class CIFAR100Corrupt(VisionDataset):
    
    base_folder = "cifar-100-corrupt"
    
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
        img_path = os.path.join(root, self.base_folder, "img.bin")
        target_path = os.path.join(root, self.base_folder, "targets.bin")
        
        with open(img_path, "rb") as f1:
            self.data = pickle.load(f1)
        with open(target_path, "rb") as f2:
            self.targets = pickle.load(f2)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            
        return index, img, target
    
    def __len__(self):
        return len(self.data)
        