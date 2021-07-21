import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import albumentations as A


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.MotionBlur(p=0.3),
            A.ElasticTransform(p=0.5, alpha=240, sigma=240 * 0.05, alpha_affine=240 * 0.03)
        ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = np.transpose(image, (1, 2, 0))
        
        transformed = self.transform(image=image, mask=label)
        
        image, label = transformed['image'], transformed['mask']
        
        
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = image.permute(2, 0, 1).contiguous()
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['arr_0'], data['arr_1']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['images'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
