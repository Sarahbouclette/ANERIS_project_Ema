## -- Transform function/ data *amplification*

import torchvision.transforms.v2 as tr
import torchvision.transforms.v2.functional as trf
import torch
import numpy as np
from PIL import Image, ImageOps
import torch.nn as nn
import torch.nn.functional as F
import os

def transform_vignettes_valid(img, data_augmentation=True):
    # convert to tensor = faster
    img = trf.to_image(img)
    
    # crop bottom
    d,h,w = img.size()
    img = trf.crop(img, 0, 0, h-31, w)

    # center on object
    img = trf.invert(img)
    sum_col = torch.sum(img[0,], 0)
    sum_row = torch.sum(img[0,], 1)

    obj_col = np.where(sum_col > 0)
    min_col = np.min(obj_col)
    max_col = np.max(obj_col)

    obj_row = np.where(sum_row > 0)
    min_row = np.min(obj_row)
    max_row = np.max(obj_row)

    w = max_col - min_col + 1
    h = max_row - min_row + 1
    img = trf.crop(img, min_row, min_col, h, w)
    h, w = img.shape[-2], img.shape[-1]

    # pad with black
    pad = np.abs((h - w) / 2)
    pad_1 = int(pad + 0.5)
    pad_2 = int(pad)
    if w < h:
        img = trf.pad(img, padding=(pad_1, 0, pad_2, 0), fill=0)
    else:
        img = trf.pad(img, padding=(0, pad_1, 0, pad_2), fill=0)

    # pipeline de transformation
    if data_augmentation:
        transform = tr.Compose([
            tr.Grayscale(num_output_channels=3),
            tr.RandomResizedCrop(224, scale=(1, 1.4), ratio=(1, 1)),
            tr.RandomRotation(90, fill=0),
            tr.RandomVerticalFlip(),
            tr.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0),
            tr.ToDtype(torch.float32, scale=True)
        ])
    else:
        transform = tr.Compose([
            tr.Grayscale(num_output_channels=3),
            tr.Resize(224),
            tr.ToDtype(torch.float32, scale=True)
        ])

    img = transform(img)
    return img

## Create Dataset PyTorch

from torch.utils.data import Dataset
from torchvision.io import read_image

class EcoTaxaDataset(Dataset):
    def __init__(self, paths, obj_ids, transform=None):
        self.paths = paths
        self.obj_ids = obj_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = read_image(self.paths[idx])
        obj_id = self.obj_ids[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, obj_id

## -- Focal loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

## -- Filter existing files
def remove_missing_or_empty_images(paths, labels):
    """
    Filter non existing or empty files.
    Args:
        paths (list of str): Path to the files
        labels (list): Coresponding labels
    Returns:
        valid_paths, valid_labels: filtered lists
    """
    valid = [(p, l) for p, l in zip(paths, labels) 
             if os.path.exists(p) and os.path.getsize(p) > 0]
    if valid:
        valid_paths, valid_labels = zip(*valid)
    else:
        valid_paths, valid_labels = [], []
    print(f"✅ {len(valid_paths)} fichiers valides sur {len(paths)}")
    return list(valid_paths), list(valid_labels)