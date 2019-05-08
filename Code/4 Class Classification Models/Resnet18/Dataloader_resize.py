from skimage import transform
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
import pickle
import pydicom
from numpy import random
import numpy as np
from torchvision import transforms

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([2048,2048]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([2048,2048]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class CBISDDSMDataset(Dataset):
    
    def __init__(self, data_file, root_dir, image_column, num_channel=1, transform = None, transform_type = 'Custom', transform_prob=0.5):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column: column name from csv file where we take the file path
        """
        self.data_frame = pickle.load(open(os.path.join(root_dir,data_file),"rb"))
        self.root_dir = root_dir
        self.transform = transform
        self.image_column = image_column
        self.num_channel = num_channel
        self.transform_prob = transform_prob
        self.transform_type = transform_type

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_frame.loc[idx, self.image_column]))
        image = pydicom.dcmread(img_name).pixel_array
        if self.num_channel > 1:
            image = np.uint8(image/65535*255)
            image = np.repeat(image[...,None],self.num_channel,axis=-1)
            image[:,:,0] = image[:,:,0]/0.11
            image[:,:,1] = image[:,:,1]/0.59
            image[:,:,2] = image[:,:,2]/0.3
        else:
            h,w = image.shape
            resized_h = 1024
            resized_w = int(resized_h/h*w)
            image = transform.resize(image, (resized_h, resized_w), anti_aliasing=True,mode='constant')
            pad_col = resized_h-resized_w
            image = np.pad(image,((0,0),(0,pad_col)),mode='constant',constant_values=0)
            image = (image - image.mean()) / image.std()
            image = image[None,...]
        
        image_class = self.data_frame.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)
        elif self.transform_type == 'Custom':
            p1 = random.uniform(0, 1)
            p2 = random.uniform(0, 1)
            if p1 <= self.transform_prob:
                image = image[:,:,-1].copy()
            if p2 <= self.transform_prob:
                image = transform.rotate(image,180)
        
        sample = {'x': image, 'y': image_class}

        return sample