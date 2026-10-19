import json
import os
import torch
import pandas as pd
from monai.transforms import (
    Compose, NormalizeIntensityd, RandZoomd,
    Resized, ToTensord, LoadImaged, EnsureChannelFirstd, Lambdad,
)
from torch.utils.data import Dataset


class QaTaUNet(Dataset):

    def __init__(self, csv_path=None, root_path=None, mode='train', image_size=[224, 224]):

        super(QaTaUNet, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f, sep=',')
        self.image_list = list(self.data['Image'])

        # ------ Train/Valid split ------
        if mode == 'train':
            self.image_list = self.image_list[:int(0.8 * len(self.image_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8 * len(self.image_list)):]
        # mode == 'test' → do nothing

        self.root_path = root_path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image_path = os.path.join(self.root_path, 'images', self.image_list[idx].replace('mask_', ''))
        mask_path  = os.path.join(self.root_path, 'masks', self.image_list[idx])

        data = {'image': image_path, 'gt': mask_path}
        data = trans(data)

        image, gt = data['image'], data['gt']

        # Convert mask values: 255 → 1
        gt = torch.where(gt == 255, 1.0, 0.0).float()

        return image, gt

    def transform(self, image_size=[224, 224]):

        if self.mode == 'train':  
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(
                    ['image', 'gt'], 
                    min_zoom=0.95, max_zoom=1.2,
                    mode=["bicubic", "nearest"], 
                    prob=0.1
                ),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])
        
        else: 
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])

        return trans
    
class MosMedUNet(Dataset):

    def __init__(self, csv_path=None, root_path=None, mode='train', image_size=[224, 224]):

        super(MosMedUNet, self).__init__()
        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f, sep=';')
        self.image_list = list(self.data['Image'])

        # ------ Train/Valid split ------
        if mode == 'train':
            self.image_list = self.image_list[:int(0.8 * len(self.image_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8 * len(self.image_list)):]
        # mode == 'test' → do nothing

        self.root_path = root_path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image_path = os.path.join(self.root_path, 'images', self.image_list[idx].replace('mask_', ''))
        mask_path  = os.path.join(self.root_path, 'masks', self.image_list[idx])

        data = {'image': image_path, 'gt': mask_path}
        data = trans(data)

        image, gt = data['image'], data['gt']

        # Convert mask values: 255 → 1
        gt = torch.where(gt == 255, 1.0, 0.0).float()

        return image, gt

    def transform(self, image_size=[224, 224]):

        if self.mode == 'train':  
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Lambdad(["image"], lambda x: x.mean(dim=0, keepdim=True)),
                RandZoomd(
                    ['image', 'gt'], 
                    min_zoom=0.95, max_zoom=1.2,
                    mode=["bicubic", "nearest"], 
                    prob=0.1
                ),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])
        
        else: 
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Lambdad(["image"], lambda x: x.mean(dim=0, keepdim=True)),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])

        return trans

class MoNuSegUNet(Dataset):

    def __init__(self, csv_path=None, root_path=None, mode='train', image_size=[224, 224]):

        super(MoNuSegUNet, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f, sep=';')
        self.image_list = list(self.data['Image'])

        self.root_path = root_path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image_path = os.path.join(self.root_path, 'images', self.image_list[idx].replace('mask_', ''))
        mask_path  = os.path.join(self.root_path, 'masks', self.image_list[idx])

        data = {'image': image_path, 'gt': mask_path}
        data = trans(data)

        image, gt = data['image'], data['gt']

        # Convert mask values: 255 → 1
        gt = torch.where(gt == 255, 1.0, 0.0).float()

        return image, gt

    def transform(self, image_size=[224, 224]):

        if self.mode == 'train':  
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Lambdad(["image"], lambda x: x.mean(dim=0, keepdim=True)),
                RandZoomd(
                    ['image', 'gt'], 
                    min_zoom=0.95, max_zoom=1.2,
                    mode=["bicubic", "nearest"], 
                    prob=0.1
                ),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])
        
        else: 
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                # convert to gray scale (1 channel)
                Lambdad(["image"], lambda x: x.mean(dim=0, keepdim=True)),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])

        return trans