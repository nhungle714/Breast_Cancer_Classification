import numpy as np
import torch.nn as nn
from torchvision import models

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from Dataloader_resize import *

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_df_path = 'Train.pkl'
val_df_path = 'Val.pkl'
test_df_path = 'Test.pkl'
root_dir = '/scratch/bva212/breastCancerData'
image_column = 'image file path'

transformed_dataset = {'train': CBISDDSMDataset(train_df_path,root_dir,image_column,num_channel=3,transform=train_transform,transform_type=None),
                       'validate':CBISDDSMDataset(val_df_path,root_dir,image_column,num_channel=3,transform=validation_transform,transform_type=None),
                       'test':CBISDDSMDataset(test_df_path,root_dir,image_column,num_channel=3,transform=validation_transform,transform_type=None)}


bs = 4

dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model.to(device)
model.load_state_dict(torch.load('/home/rc3620/BMSC-GA-4493/Project/Transfer_Learning/Model_try1_Wt'))

label_all = np.zeros((1,4))
output_all = np.zeros((1,4))
patient_num = []
running_correct = 0
running_total = 0
for data in dataloader['test']:
    model.eval()
    with torch.no_grad():
        image = torch.as_tensor(data['x'],dtype=torch.float,device=device)
        label = torch.as_tensor(data['y'],dtype=torch.long,device=device)
        output = model(image)
        _, preds = torch.max(output, dim = 1)
        output1 = torch.nn.functional.softmax(output,dim=1)
        num_imgs = image.size()[0]
        running_correct += torch.sum(preds ==label).item()
        running_total += num_imgs
        y_onehot = torch.zeros(label.shape[0], 4).scatter_(1, label.cpu().unsqueeze(1), 1)
        label_all = np.concatenate([label_all,y_onehot.numpy()],axis=0)
        output_all = np.concatenate([output_all,output1.cpu().detach().numpy()],axis=0)
        patient_num.append(data['pid'])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(label_all[1:, i], output_all[1:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(label_all.ravel(), output_all.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

from scipy import interp
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(4):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 4

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(roc_auc)
acc = float(running_correct/running_total)
print(acc)

len_pid = len(patient_num)
p_id_index_sort = np.array(patient_num).reshape(len_pid).argsort()
label_new = label_all[1:]
label_all_sort = label_new[p_id_index_sort]
output_new = output_all[1:]
output_all_sort = output_new[p_id_index_sort]
sorted_patient_list = np.sort(np.array(patient_num).reshape(300))

len_list = len(sorted_patient_list)
prob = np.exp(output_all_sort)/np.sum(np.exp(output_all_sort),axis=1).reshape(300,1)
patient_actual_label = np.zeros_like(label_all_sort)
patient_pred_label = np.zeros_like(output_all_sort)
i=0
while(i<len_list):
    patient_actual_label[i] = label_all_sort[i]
    if i < len_list-1 and sorted_patient_list[i] == sorted_patient_list[i+1]:
        patient_pred_label[i] = np.max(prob[i:i+2],axis=0)
        patient_pred_label[i] = patient_pred_label[i]/sum(patient_pred_label[i])
        i += 2
    else:
        patient_pred_label[i] = prob[i]
        i += 1

patient_pred_label_nz = patient_pred_label[np.sum(patient_actual_label,axis=1)!=0]
patient_act_label_nz = patient_actual_label[np.sum(patient_actual_label,axis=1)!=0]

# Compute ROC curve and ROC area for each class
p_fpr = dict()
p_tpr = dict()
p_roc_auc = dict()
for i in range(4):
    p_fpr[i], p_tpr[i], _ = roc_curve(patient_act_label_nz[:, i], patient_pred_label_nz[:, i])
    p_roc_auc[i] = auc(p_fpr[i], p_tpr[i])

# Compute micro-average ROC curve and ROC area
p_fpr["micro"], p_tpr["micro"], _ = roc_curve(label_all.ravel(), output_all.ravel())
p_roc_auc["micro"] = auc(p_fpr["micro"], p_tpr["micro"])

from scipy import interp
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([p_fpr[i] for i in range(4)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(4):
    mean_tpr += interp(all_fpr, p_fpr[i], p_tpr[i])

# Finally average it and compute AUC
mean_tpr /= 4

p_fpr["macro"] = all_fpr
p_tpr["macro"] = mean_tpr
p_roc_auc["macro"] = auc(p_fpr["macro"], p_tpr["macro"])