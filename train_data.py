"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train_data.py
about: build the training dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
import torch
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import glob
import h5py
import numpy as np

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.imgs = glob.glob(self.train_data_dir+'/*.h5')

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        
        path = self.imgs[index]
        f = h5py.File(path,'r')

        haze_img = f['haze'][:]
        gt_img = f['gt'][:]

#         gt_img = np.swapaxes(gt_img,0,2)
#         haze_img = np.swapaxes(haze_img,0,2)
        width, height, _ = haze_img.shape

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
#         haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        haze_crop_img = haze_img[x:x+crop_width, y:y+crop_height,:]
#         gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img[x:x+crop_width, y:y+crop_height,:]

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        
        haze = haze.type(torch.FloatTensor)
        gt = gt.type(torch.FloatTensor)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.imgs)

