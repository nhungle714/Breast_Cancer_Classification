import pandas as pd
from sklearn.utils import shuffle
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from skimage import io
import torch
from torchvision import transforms
import torchvision
from skimage import color
import copy

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage import io, transform
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import scipy
import random
import pickle
import scipy.io as sio
import itertools
from scipy.ndimage.interpolation import shift
import copy
import warnings
#warnings.filterwarnings("ignore")
plt.ion()
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

excel_path = '/Users/nhungle/Box/Free/Data-Science-Projects/Breast_Cancer_Diagnosis/excel_files'

cal_test = pd.read_csv(os.path.join(excel_path, 'calc_case_description_test_set.csv'))
cal_train = pd.read_csv(os.path.join(excel_path, 'calc_case_description_train_set.csv'))
mass_test = pd.read_csv(os.path.join(excel_path, 'mass_case_description_test_set.csv'))
mass_train = pd.read_csv(os.path.join(excel_path, 'mass_case_description_train_set.csv'))

cal_test = cal_test[['patient_id','left or right breast', 'image view',
            'pathology', 'image file path',
        'cropped image file path', 'ROI mask file path']]
mass_test = mass_test[['patient_id','left or right breast', 'image view',
            'pathology', 'image file path',
        'cropped image file path', 'ROI mask file path']]
cal_train = cal_train[['patient_id','left or right breast', 'image view',
        'pathology', 'image file path',
    'cropped image file path', 'ROI mask file path']]
mass_train = mass_train[['patient_id','left or right breast', 'image view',
        'pathology', 'image file path',
    'cropped image file path', 'ROI mask file path']]

cal_test.dropna(axis = 0, how = 'all', inplace = True)
mass_test.dropna(axis = 0, how = 'all', inplace = True)
testSet = pd.concat([cal_test, mass_test], axis=0)
train_valid = pd.concat([cal_train, mass_train], axis=0)

train_valid['class'] = 0
train_valid.loc[train_valid['pathology'] == 'MALIGNANT', 'class'] = 1
testSet['class'] = 0
testSet.loc[testSet['pathology'] == 'MALIGNANT', 'class'] = 

patient_list = list(train_valid['patient_id'].unique())
print('Number of patients: {}'.format(len(patient_list)))

unique_id_leftright = train_valid.groupby(['patient_id', 'left or right breast']).size().reset_index().rename(columns={0:'count'})
unique_id_leftright = shuffle(unique_id_leftright)
train_size = int(len(unique_id_leftright) * 0.8)
#print(train_size)
train_id_leftright = unique_id_leftright[:train_size][['patient_id', 'left or right breast']]
valid_id_leftright = unique_id_leftright[train_size:][['patient_id', 'left or right breast']]

trainSet = train_valid[(train_valid['patient_id'].isin(train_id_leftright['patient_id'])) &
            (train_valid['left or right breast'].isin(train_id_leftright['left or right breast'])) ]

validSet = train_valid[(train_valid['patient_id'].isin(valid_id_leftright['patient_id'])) &
            (train_valid['left or right breast'].isin(valid_id_leftright['left or right breast'])) ]

# Write to CSV
trainSet.to_csv(os.path.join(excel_path, 'twoClass_trainSet.csv'))
validSet.to_csv(os.path.join(excel_path, 'twoClass_validSet.csv'))
testSet.to_csv(os.path.join(excel_path, 'twoClass_testSet.csv'))

# Get a subset of data 
trainSet_sample = trainSet[:500]
valSet_sample = validSet[:100]
testSet_sample = testSet[:100]

trainSet_sample.to_csv(os.path.join(excel_path, 'twoClass_trainSet_sample.csv'))
valSet_sample.to_csv(os.path.join(excel_path, 'twoClass_validSet_sample.csv'))
testSet_sample.to_csv(os.path.join(excel_path, 'twoClass_testSet_sample.csv'))