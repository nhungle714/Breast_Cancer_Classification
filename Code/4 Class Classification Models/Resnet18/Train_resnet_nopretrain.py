from torch import optim
from Dataloader import *
import time 
from sklearn.metrics import roc_curve, auc
from resnet import *

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_df_path = 'Train.pkl'
val_df_path = 'Val.pkl'
test_df_path = 'Test.pkl'
root_dir = '/scratch/bva212/breastCancerData'
image_column = 'image file path'
transformed_dataset = {'train': CBISDDSMDataset(train_df_path,root_dir,image_column),
                       'validate':CBISDDSMDataset(val_df_path,root_dir,image_column),
                       'test':CBISDDSMDataset(test_df_path,root_dir,image_column)}
                       
bs = 4

num_classes = 4

dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}

def train_model(model, optimizer, scheduler, loss_fn, num_epochs = 10):
    acc_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    best_auc = 0
    phases = ['train','validate']
    since = time.time()
    for i in range(num_epochs):
        print('Epoch: {}/{}'.format(i, num_epochs-1))
        print('-'*10)
        for p in phases:
            running_correct = 0
            running_loss = 0
            running_total = 0
            pred_all = np.zeros((1,4))
            label_all = np.zeros((1,4))
            for data in dataloader[p]:
                optimizer.zero_grad()
                if p == 'train':
                    model.train()
                else:
                    model.eval()
                with torch.set_grad_enabled(p=='train'):
                    image = torch.as_tensor(data['x'],dtype=torch.float,device=device)
                    label = torch.as_tensor(data['y'],dtype=torch.long,device=device)
                    output = model(image)
                    loss = loss_fn(output, label)
                    _, preds = torch.max(output, dim = 1)
                    num_imgs = image.size()[0]
                    running_correct += torch.sum(preds ==label).item()
                    running_loss += loss.item()
                    running_total += num_imgs
                    if p == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        y_onehot = torch.zeros(label.shape[0], num_classes).scatter_(1, label.cpu().unsqueeze(1), 1)
                        label_all = np.concatenate([label_all,y_onehot.numpy()],axis=0)
                        pred_all = np.concatenate([pred_all,output.cpu().detach().numpy()],axis=0)
            epoch_acc = float(running_correct/running_total)
            epoch_loss = float(running_loss/running_total)
            acc_dict[p].append(epoch_acc)
            loss_dict[p].append(epoch_loss)
            if p == 'validate':
                fpr, tpr, _ = roc_curve(label_all.ravel(), pred_all.ravel())
                roc_auc = auc(fpr, tpr)
                scheduler.step()
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_model_wts = model.state_dict()
                print('Phase:{}, epoch loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(p, epoch_loss, epoch_acc, roc_auc))
            else:
                print('Phase:{}, epoch loss: {:.4f} Acc: {:.4f}'.format(p, epoch_loss, epoch_acc))
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val auc: {:4f}'.format(best_auc))
    
    checkpoint_mdl_wt = model.state_dict()
    checkpoint_optim = optimizer.state_dict()
    checkpoint_scheduler = scheduler.state_dict()
    model.load_state_dict(best_model_wts)
    return model, checkpoint_mdl_wt, checkpoint_optim, checkpoint_scheduler, acc_dict, loss_dict


model = ResNet(BasicBlock,[2,2,2,2]).to(device)
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 8)
model, checkpoint_mdl_wt, checkpoint_optim, checkpoint_scheduler, acc_dict, loss_dict = train_model(model,optimizer, scheduler, loss_fn,num_epochs =30)

torch.save(model.state_dict(), 'Model_resnet_Wt')
torch.save(checkpoint_mdl_wt,'Model_resnet_chkpt_Wt')
torch.save(checkpoint_optim,'Model_resnet_chkpt_optim')
torch.save(checkpoint_scheduler,'Model_resnet_chkpt_scheduler')
pickle.dump(acc_dict,open("Model_resnet_AccDict.pkl","wb"))
pickle.dump(loss_dict,open("Model_resnet_LossDict.pkl","wb"))