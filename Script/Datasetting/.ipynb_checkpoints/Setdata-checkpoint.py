
import torchvision.transforms.v2 as tr
import torchvision.transforms.v2.functional as trf
import torch
import numpy as np
from PIL import Image, ImageOps
import torch.nn as nn
import torch.nn.functional as F
import os

## -- Transform function/ data *amplification*
def transform_vignettes_valid(img, data_augmentation=True):
    # convert to tensor = faster
    img = trf.to_image(img)
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

## -- Create Dataset PyTorch
from torch.utils.data import Dataset
from torchvision.io import read_image

class EcoTaxaDataset(Dataset):
    def __init__(self, paths, labels=None, class_to_idx=None, idx_to_class=None, transform=None):
      self.paths = paths
      self.labels = labels
      self.transform = transform
      self.class_to_idx = class_to_idx
      self.idx_to_class = idx_to_class

    def __len__(self):
      return len(self.paths)

    def __getitem__(self, idx):
      img = read_image(self.paths[idx])
      if self.transform:
        img = self.transform(img)
      if self.labels is None:
        label = -1
      else:
        label = self.class_to_idx[self.labels[idx]]
      return img,label

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
