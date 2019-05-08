from skimage import transform
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
import pickle
import pydicom
from numpy import random
import numpy as np

def last_nonzero(arr, axis):
    mask = np.sum(arr,axis=1-axis)!=0
    val = arr.shape[axis] - np.flip(mask).argmax() - 1
    return val

def first_nonzero(arr, axis):
    mask = np.sum(arr,axis=1-axis)!=0
    return mask.argmax()


class CBISDDSMDataset(Dataset):
    
    def __init__(self, data_file, root_dir, image_column, roi_column, num_channel=1, transform = None, transform_type = 'Custom', transform_prob=0.5):
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
        self.roi_column = roi_column
        self.num_channel = num_channel
        self.transform_prob = transform_prob
        self.transform_type = transform_type

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_frame.loc[idx, self.image_column]))
        image = pydicom.dcmread(img_name).pixel_array
        roi_img_path = os.path.join(self.root_dir, str(self.data_frame.loc[idx, self.roi_column])[:-10]+'000001.dcm')
        if os.path.isfile(roi_img_path) and os.path.getsize(roi_img_path)>10**6:
            roi_img_path = roi_img_path
        else:
            roi_img_path = os.path.join(self.root_dir, str(self.data_frame.loc[idx, self.roi_column][:-10]+'000000.dcm'))
        roi_image = pydicom.dcmread(roi_img_path).pixel_array

        shape0 = image.shape
        shape1 = roi_image.shape
        h,w = image.shape
        pad_row = 7500-h
        pad_col = 5500-w
        
        if shape0 == shape1:
            if sum(image[:,-1]) == 0:
                image = np.pad(image,((0,pad_row),(0,pad_col)),mode='constant',constant_values=0)
                roi_image = np.pad(roi_image,((0,pad_row),(0,pad_col)),mode='constant',constant_values=0)
            else:
                image = np.pad(image,((0,pad_row),(pad_col,0)),mode='constant',constant_values=0)
                roi_image = np.pad(roi_image,((0,pad_row),(pad_col,0)),mode='constant',constant_values=0)

            image = np.float32(image/np.iinfo(image.dtype).max)
            roi_image = np.float32(roi_image/np.iinfo(roi_image.dtype).max)
            shape = roi_image.shape

            image = (image - 0.3328) / 0.7497
            if self.num_channel > 1:
                image=np.repeat(image[None,...],self.num_channel,axis=0)

            image_class = self.data_frame.iloc[idx, -1]

            if self.transform:
                image = self.transform(image)
                roi_image = self.transform(roi_image)
            elif self.transform_type == 'Custom':
                p1 = random.uniform(0, 1)
                p2 = random.uniform(0, 1)
                if p1 <= self.transform_prob:
                    image = np.flip(image,0).copy()
                    roi_image = np.flip(roi_image,0).copy()
                if p2 <= self.transform_prob:
                    image = np.flip(image,1).copy()
                    roi_image = np.flip(roi_image,1).copy()


            x1 = first_nonzero(roi_image,0)/shape1[0]
            x2 = last_nonzero(roi_image,0)/shape1[0]
            y1 = first_nonzero(roi_image,1)/shape1[1]
            y2 = last_nonzero(roi_image,1)/shape1[1]
            sample = {'x': image[None,...],'y': image_class, 'b': np.array([x1,x2,y1,y2])}
        
        else:
            sample = None

        return sample