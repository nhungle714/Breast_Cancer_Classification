from torchvision import models
from torch import optim
from torch import nn
from Dataloader import *
import time 

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_df_path = 'Train.pkl'
val_df_path = 'Val.pkl'
test_df_path = 'Test.pkl'
root_dir = '/scratch/bva212/breastCancerData'
image_column = 'image file path'
transformed_dataset = {'train': CBISDDSMDataset(train_df_path,root_dir,image_column,num_channel=3),
                       'validate':CBISDDSMDataset(val_df_path,root_dir,image_column,num_channel=3,transform_type=None),
                       'test':CBISDDSMDataset(test_df_path,root_dir,image_column,num_channel=3,transform_type=None)}
                       
bs = 1

dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}

def train_func(optimizer,data,model,loss_fn, running_correct, running_loss, running_total):
    optimizer.zero_grad()
    image = torch.as_tensor(data['x'],dtype=torch.float,device=device)
    label = torch.as_tensor(data['y'],dtype=torch.long,device=device)
    output = model(image)
    loss = loss_fn(output, label)
    _, preds = torch.max(output, dim = 1)
    num_imgs = image.size()[0]
    running_correct += torch.sum(preds ==label).item()
    running_loss += loss.item()
    running_total += num_imgs
    loss.backward()
    optimizer.step()
    return running_correct, running_loss, running_total

def eval_func(optimizer,data,model,loss_fn, running_correct, running_loss, running_total):
    with torch.no_grad():
        image = torch.as_tensor(data['x'],dtype=torch.float,device=device)
        label = torch.as_tensor(data['y'],dtype=torch.long,device=device)
        output = model(image)
        loss = loss_fn(output, label)
        _, preds = torch.max(output, dim = 1)
        num_imgs = image.size()[0]
        running_correct += torch.sum(preds ==label).item()
        running_loss += loss.item()
        running_total += num_imgs
        return running_correct, running_loss, running_total

def train_model(model, optimizer, loss_fn, num_epochs = 10):
    acc_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    best_acc = 0
    phases = ['train','validate']
    since = time.time()
    for i in range(num_epochs):
        print('Epoch: {}/{}'.format(i, num_epochs-1))
        print('-'*10)
        for p in phases:
            running_correct = 0
            running_loss = 0
            running_total = 0                
            for data in dataloader[p]:
                if p == 'train':
                    model.train()
                    running_correct, running_loss, running_total = train_func(optimizer,data,model,loss_fn, running_correct, running_loss, running_total)
                else:
                    model.eval()
                    running_correct, running_loss, running_total = eval_func(optimizer,data,model,loss_fn, running_correct, running_loss, running_total)
                    
            epoch_acc = float(running_correct/running_total)
            epoch_loss = float(running_loss/running_total)
            print(torch.cuda.memory_allocated(device=device)/1024/1024,"MB")
            print('Phase:{}, epoch loss: {:.4f} Acc: {:.4f}'.format(p, epoch_loss, epoch_acc))
            
            acc_dict[p].append(epoch_acc)
            loss_dict[p].append(epoch_loss)
            if p == 'validate':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    
    checkpoint_mdl_wt = model.state_dict()
    checkpoint_optim = optimizer.state_dict()
    
    model.load_state_dict(best_model_wts)
    return model, checkpoint_mdl_wt, checkpoint_optim, acc_dict, loss_dict
    
# model = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
# model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
# model.last_linear = nn.Linear(in_features=1536, out_features=4)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model.to(device)

print(torch.cuda.memory_allocated(device=device)/1024/1024,"MB")
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, checkpoint_mdl_wt, checkpoint_optim, acc_dict, loss_dict = train_model(model,optimizer,loss_fn,num_epochs =5)

torch.save(model.state_dict(), 'Model_try1_Wt')
torch.save(checkpoint_mdl_wt,'Model_try1_chkpt_Wt')
torch.save(checkpoint_optim,'Model_try1_chkpt_optim')
pickle.dump(acc_dict,open("Model_Try1_AccDict.pkl","wb"))
pickle.dump(loss_dict,open("Model_Try1_LossDict.pkl","wb"))

