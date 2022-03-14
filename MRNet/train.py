import torch
import time
import copy


# Training model and validation 
def train_model(model, dataloaders, device, criterion, optimizer, train_writer, \
                train_loss_writer, valid_writer, valid_loss_writer, scheduler,  \
                num_epochs, n_epochs_stop, model_path, model_path2, is_inception=False):

    since = time.time()
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_iteration = 0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts_loss = copy.deepcopy(model.state_dict())

    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(scheduler.get_lr())
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':    
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
 
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'train':
                train_writer.add_scalar('Train Accuracy', epoch_acc, epoch)
                train_loss_writer.add_scalar('Train Loss', epoch_loss, epoch)
                if epoch_acc > best_train_acc:
                    best_train_acc = epoch_acc      

            if phase == 'val':
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_iteration = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts_loss = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1
                valid_writer.add_scalar('Valid Accuracy', epoch_acc, epoch)
                valid_loss_writer.add_scalar('Valid Loss', epoch_loss, epoch)
                
        scheduler.step()
        print()
        
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            break
        else:
            continue
            
    train_writer.close()
    train_loss_writer.close()
    valid_writer.close()
    valid_loss_writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {}'.format(best_iteration))  
    print('Best train Acc: {:4f}'.format(best_train_acc)) 
    print('Best val Acc: {:4f}'.format(best_val_acc))

    # Save and load best model weights
    torch.save(best_model_wts, model_path)
    torch.save(best_model_wts_loss, model_path2)