import torch
import argparse
import time
import copy
import neptune
from torchvision import transforms

class FullTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(FullTrainingDataset, self).__init__()
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        return self.full_ds[i+self.offset]
    
def trainTestSplit(dataset, val_share):
    val_offset = int(len(dataset)*val_share)
    return FullTrainingDataset(dataset, 0, val_offset), FullTrainingDataset(dataset, val_offset, len(dataset)-val_offset)

def train_model(opt, net, dataset):
    model = net.model
    criterion = net.criterion
    optimizer = net.optimizer
    dataloaders = dataset.dataloaders
    dataset_sizes = dataset.dataset_sizes

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(opt.epochs):
        print('Epoch {}/{}'.format(epoch, opt.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
#                 print(idx, len(dataloaders[phase]))
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if opt.sendNeptune:
                neptune.send_metric('{}_loss'.format(phase), epoch_loss)
                neptune.send_metric('{}_acc'.format(phase), epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} achieved on model from epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    if opt.sendNeptune:
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, "./best_model.pth")
        neptune.send_artifact('best_model.pth')
    return model


def getConfig():
    # naprawic to tak, zeby nie byly potrzebne dwa parsery.
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--dataset', required=True, help='Cifar | ImageNet')
    parser.add_argument('--batchSize', type=int, required = True)
    parser.add_argument('--sessionName', required = True)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--net', )
    parser.add_argument('--cifarFactor', type=float, required = True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weightDecay', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpuId', type=int, required=True)

    parserShell = argparse.ArgumentParser()
    parserShell.add_argument('--sendNeptune', action='store_true')
    parserShell.add_argument('--pretrained', action='store_true')

    opt = parser.parse_args(['@config.txt'])
    optShell = parserShell.parse_args()

    opt.sendNeptune = optShell.sendNeptune
    opt.pretrained = optShell.pretrained
    opt.device = torch.device("cuda:{}".format(opt.gpuId) if torch.cuda.is_available() else "cpu")
    if opt.dataset == "Cifar":
        opt.outSize = 10
    else:
        opt.outSize = 100

    opt.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return opt
