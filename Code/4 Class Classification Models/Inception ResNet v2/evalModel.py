import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.dataloader import *
import pickle
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def evaluate_model(model, dataloader, loss_fn, phase = 'test'):
    model.eval()
    running_correct = 0
    running_loss = 0
    running_total = 0
    outputs = np.array(0)
    preds = np.array(0)
    labels = np.array(0)
    for data in dataloader[phase]:
        inputs = data['x'].type(torch.FloatTensor).to(device)
        label = data['y'].to(device)
        output = model(inputs)
        loss = loss_fn(output, label)
        _, pred = torch.max(output, dim = 1)
        num_inputs = inputs.size()[0]
        outputs = np.append(outputs, output.cpu().detach().numpy())
        preds = np.append(preds, pred.cpu().detach().numpy())
        labels = np.append(labels, label.cpu().detach().numpy())
        running_correct += torch.sum(pred ==label).item()
        running_loss += loss.item()*num_inputs
        running_total += num_inputs
    accuracy = float(running_correct/running_total)
    loss = float(running_loss/running_total)
    
    return outputs[1:], preds[1:], labels[1:], accuracy, loss

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', 
                          root_dir = '/scratch/bva212/dl4medProject/', model_folder = '', 
                          cmap=None, normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.legend()
    plt.savefig(os.path.join(root_dir,model_folder, 'ConfusionMatrix - Test'))
    plt.show()