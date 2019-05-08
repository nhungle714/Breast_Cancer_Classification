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
import pydicom

class MammogramDataset_Custom(Dataset):   
    def __init__(self, csv_file, root_dir, image_column, num_channel=1, transform = None, 
                 transform_type = 'Custom', transform_prob=0.5):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column: column name from csv file where we take the file path
        """
        #self.data_frame = pickle.load(open(os.path.join(root_dir,data_file),"rb"))
        self.data_frame = pd.read_csv(csv_file)
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
        
        image_class = self.data_frame.loc[idx, 'class']

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
            
        
        sample = {'x': image[None,:], 'y': image_class}
        return sample


def GetDataLoader_TL(train_csv, validation_csv, test_csv, 
                     root_dir, image_column, num_channel, 
                     transform_type, transform_prob, 
               train_transform, validation_transform, 
               batch_size, shuffle, num_workers): 

    train_data = MammogramDataset_Custom(csv_file = train_csv, 
                              root_dir = root_image,
                              image_column = image_column,
                              num_channel = num_channel, 
                               transform=train_transform, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    val_data = MammogramDataset_Custom(csv_file = validation_csv, 
                            root_dir = root_image,
                            image_column = image_column,
                            transform = validation_transform,
                                 num_channel = num_channel, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    test_data = MammogramDataset_Custom(csv_file = test_csv, 
                            root_dir = root_image,
                            image_column = image_column,
                            transform = validation_transform,
                            num_channel = num_channel, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    
    image_datasets = {'train': train_data, 'val': val_data, 'test': test_data}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True, 
                                              num_workers=NUM_WORKERS) 
                    for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    return dataloaders, dataset_sizes



def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs = 10,verbose = True):

    start_time = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    acc_dict = {'train':[],'validation':[]}
    loss_dict = {'train':[],'validation':[]}
    auc_dict ={'train': [], 'validation': []}

    for epoch in range(num_epochs):
        if verbose:
            #if epoch % 5 == 4:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
            
        ## Scheduler learning step
        #scheduler.step()
            
        for phase in ['train','val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            whole_output = []
            whole_target = []

            for data in dataloaders[phase]:
                
                inputs = data['x']
                labels = data['y']
                
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.to(device)
                else:
                    inputs = Variable(inputs).type(torch.FloatTensor)
                    labels = Variable(labels).type(torch.LongTensor)

                    
                optimizer.zero_grad()

                out = model(inputs)
                _, preds = torch.max(out, dim = 1)
                loss = criterion(out, labels)
                
                #To get AUC score later
                output =F.softmax(model(inputs),dim=1)
                whole_output.append( output.cpu().data.numpy())
                whole_target.append( data['y'])
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * inputs.size()[0]
                running_corrects += torch.sum(preds == labels).item()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_whole_output = np.concatenate(whole_output)
            y_target = list(np.concatenate(whole_target))
            y_score = [output[1] for output in epoch_whole_output]
            fpr, tpr, _ = roc_curve(y_target, y_score, pos_label=1)
            epoch_auc = auc(fpr, tpr)

            if verbose:
#                 if epoch % 5 == 4:
                print('{} Loss: {:.4f} Acc: {:.4f} Auc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_auc))

            if phase == 'train':
                loss_dict['train'].append(epoch_loss)
                acc_dict['train'].append(epoch_acc)
                auc_dict['train'].append(epoch_auc)
            else:
                loss_dict['validation'].append(epoch_loss)
                acc_dict['validation'].append(epoch_acc)
                auc_dict['validation'].append(epoch_auc)
                    
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                scheduler.step(epoch_loss)

    time_elapsed = time.time() - start_time
    print('Training time: {}minutes {}s'.format(int(time_elapsed / 60), time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(graph_path, '{}.pt'.format(model_name)))
    
    return {'Model': model, 'LossDict': loss_dict, 'AccDict': acc_dict, 'AucDict': auc_dict}

####################  Develop Custom CNN Model ###########################

class CNN_Disease(nn.Module):
    def __init__(self, out_features=2):
        super(CNN_Disease, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=2)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,32,kernel_size=3,stride=2)
        self.relu4 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.relu6 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.relu7 = nn.ReLU()
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.relu9 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64,128,kernel_size=3,stride=2)
        self.relu10 = nn.ReLU()
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(128,256,kernel_size=3,stride=2)
        self.relu11 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256,out_features)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max1(x)
        y = self.relu2(self.conv2(self.bn1(x)))
        y = self.conv3(y)
        x = self.relu3(y + x)
        x = self.relu4(self.conv4(self.bn2(x)))
        x = self.max2(x)
        y = self.relu5(self.conv5(self.bn3(x)))
        y = self.conv6(y)
        x = self.relu6(y + x)
        x = self.relu7(self.conv7(self.bn4(x)))
        x = self.max3(x)
        y = self.relu8(self.conv8(self.bn5(x)))
        y = self.conv9(y)
        x = self.relu9(y + x)
        x = self.relu10(self.conv10(self.bn6(x)))
        x = self.max4(x)
        x = self.avgpool(self.relu11(self.conv11(x)))
        x = self.fc(x.view(-1,256))
        return x

####################  Define Data Path ###########################

######### HPC Paths - Sample Data Set ######## 
excel_path = '/home/nhl256/BreastCancer/excel_files'
train_local_csv = os.path.join(excel_path, 
                             'twoClass_trainSet.csv')
validation_local_csv = os.path.join(excel_path, 
                              'twoClass_validSet.csv')
test_local_csv = os.path.join(excel_path, 
                              "twoClass_testSet.csv")


image_path = '/scratch/bva212/breastCancerData'
#root_image = os.path.join(image_path ,'CBIS-DDSM')
root_image = image_path

NUM_WORKERS = 4
BATCH_SIZE = 2
graph_path = '/home/nhl256/BreastCancer/graphs'
image_column = 'image file path'


#################### Get Dataloaders and Datasets_sizes ###########################
use_gpu = torch.cuda.is_available()
if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#### Get Dataloaders and Datasets_sizes
dataloaders, dataset_sizes = GetDataLoader_TL(train_csv = train_local_csv, 
                                            validation_csv = validation_local_csv, 
                                            test_csv = test_local_csv, 
                                            root_dir = root_image, 
                                           image_column = image_column,
                                            num_channel = 1,
                                            transform_type = None, 
                                              transform_prob=0.5,
               train_transform =None, validation_transform = None, 
               batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS) 

#################### Train Model ###########################
if use_gpu:
    model = CNN_Disease().to(device)
    model.load_state_dict(torch.load(os.path.join(excel_path, 'Custom_ModelBonus_Wt')))
else: 
    model = CNN_Disease()
    model.load_state_dict(torch.load(os.path.join(excel_path, 'Custom_ModelBonus_Wt'), map_location='cpu'))

optimizer = torch.optim.Adam(model.fc.parameters(), lr = 0.00001, weight_decay=1)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)


BestCNN = train_model(model, 'CNN_full', criterion, optimizer, scheduler, num_epochs = 20, verbose = True)


################ Plot #####################
def PlotAccLoss(model, model_name, plot_dict, plot_name): 
    fig, ax = plt.subplots()
    for key in model[plot_dict]: 
        ax.plot(model[plot_dict][key], label = key)
    ax.set_title('Train and Validation {}'.format(plot_name))
    ax.set_ylabel('{}'.format(plot_name))
    ax.set_xlabel('Epochs')
    legend = ax.legend(loc= 'best', shadow=True,
                          bbox_to_anchor = (0.5, 0, 0.5, 0.5), ncol = 1, prop = {'size': 10})
    plt.savefig(os.path.join(graph_path ,'{} Curves_{}.png'.format(plot_name, model_name)))

PlotAccLoss(BestCNN, 'Custom_CNN_Full', 'LossDict', 'Loss')
PlotAccLoss(BestCNN, 'Custom_CNN_Full', 'AccDict', 'Accuracy')
PlotAccLoss(BestCNN, 'Custom_CNN_Full', 'AucDict', 'AUC')

################ Evaluation on Test Set #####################
def inference(model_ft,loader):
    model_ft.eval()
    whole_output =[]
    whole_target = []
    

    for valData in loader:
        data = valData['x']
        target = valData['y']
        if use_gpu:
            data = Variable(data,volatile=True).type(torch.FloatTensor).cuda()
            target = Variable(target,volatile=True).type(torch.LongTensor).cuda()
        else:
            data= Variable(data,volatile=True).type(torch.FloatTensor)
            target = Variable(target,volatile=True).type(torch.LongTensor)

        output =F.softmax(model_ft(data),dim=1)
        whole_output.append( output.cpu().data.numpy())
        whole_target.append( valData['y'].numpy())

    whole_output = np.concatenate(whole_output)
    whole_target = list(np.concatenate(whole_target))
    y_target = whole_target


    #print('Whole_output: {}, whole_target: {}'.format(whole_output, whole_target))
    #print('y_target: {}'.format(y_target))
    y_score = [output[1] for output in whole_output]
    return y_score, y_target

def write_list_to_file(filename, my_list):
    with open(filename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


##### Inference Resnet18 #######
BestCNN = torch.load(os.path.join(graph_path, 'CNN_full.pt'))
y_score_CNN, y_target_CNN = inference(BestCNN, dataloaders['test'])
write_list_to_file(os.path.join(graph_path, 'y_score_CNN_full.txt'), y_score_CNN)
write_list_to_file(os.path.join(graph_path, 'y_target_CNN_full.txt'), y_target_CNN)

