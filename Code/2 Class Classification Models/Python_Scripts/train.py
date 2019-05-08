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
                #print('out: {}'.format(out))
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
            epoch_roc_auc = auc(fpr, tpr)

            if verbose:
#                 if epoch % 5 == 4:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                loss_dict['train'].append(epoch_loss)
                acc_dict['train'].append(epoch_acc)
                auc_dict['train'].append(epoch_roc_auc)
            else:
                loss_dict['validation'].append(epoch_loss)
                acc_dict['validation'].append(epoch_acc)
                auc_dict['validation'].append(epoch_roc_auc)
                    
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                scheduler.step(epoch_loss)

    time_elapsed = time.time() - start_time
    print('Training time: {}minutes {}s'.format(int(time_elapsed / 60), time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(graph_path, '{}.pt'.format(model_name)))
    
    return {'Model': model, 'LossDict': loss_dict, 'AccDict': acc_dict, 
           'AucDict': auc_dict}
