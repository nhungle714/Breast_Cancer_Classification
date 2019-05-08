from skimage import transform
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
import pickle
import pydicom
from numpy import random
import numpy as np

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
        h,w = image.shape
        pad_row = 7500-h
        pad_col = 5500-w
        if sum(image[:,-1]) == 0:
            image = np.pad(image,((0,pad_row),(0,pad_col)),mode='constant',constant_values=0)
        else:
            image = np.pad(image,((0,pad_row),(pad_col,0)),mode='constant',constant_values=0)
        image = np.float32(image/np.iinfo(image.dtype).max)

        image = (image - 0.3328) / 0.7497
        if self.num_channel > 1:
            image=np.repeat(image[None,...],self.num_channel,axis=0)
        
        image_class = self.data_frame.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)
        elif self.transform_type == 'Custom':
            p1 = random.uniform(0, 1)
            p2 = random.uniform(0, 1)
            if p1 <= self.transform_prob:
                if p2 <= self.transform_prob:
                    image = np.flip(image,0).copy()
                else:
                    image = np.flip(image,1).copy()
            
        
        sample = {'x': image[None,:], 'y': image_class, 'pid': self.data_frame.loc[idx, 'patient_id']}

        return sample