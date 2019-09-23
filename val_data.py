"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: val_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import glob
import h5py
import numpy as np

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        self.imgs = glob.glob(val_data_dir+'/*.h5')

    def get_images(self, index):
        path = self.imgs[index]
        f = h5py.File(path,'r')

        haze_img = f['haze'][:]
        gt_img = f['gt'][:]
        
#         gt_img = np.swapaxes(gt_img,0,2)
#         haze_img = np.swapaxes(haze_img,0,2)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)
        
        haze = haze.type(torch.FloatTensor)
        gt = gt.type(torch.FloatTensor)
        
        return haze, gt, path

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.imgs)
