import torch
import argparse
import time
import copy
import neptune
import os
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)
    
    def __len__(self):
        return max(len(d) for d in self.datasets)
    

def getCifar(opt):
    image_datasets = {'train': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=True, download=True, transform=transform), 
                        'val': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=False, download=True, transform=transform)}
    return image_datasets

def getImageNet(opt):
    image_datasets = {x: datasets.ImageFolder(os.path.join('~/data/lilImageNet', x),
                                                transform=transform)
                        for x in ['train', 'val']}
    return image_datasets

def getCifarUnlabeled(opt):
    image_dataset = torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=True, download=True, transform=transform)
    return image_dataset

def train_model(opt, net, dataset):
    model = net.model
    criterion = net.criterion
    optimizer = net.optimizer
    dataloaders = dataset.dataloaders
    dataset_sizes = dataset.dataset_sizes

    since = time.time()
    best_model_wts = copy.deepcopy(model.model_ft.state_dict())
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
            running_cons_loss = 0.0

            # Iterate over data.
            for idx, batches in enumerate(dataloaders[phase]):
                if not opt.onlyLabeled:
                    inputs, labels = batches[0]
                    inputs_unlabeled, _ = batches[1]
                    inputs_unlabeled = inputs_unlabeled.to(opt.device)
                else:
                    inputs, labels = batches
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs, not opt.pretrained)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if not opt.onlyLabeled:
                        outputs_unlabeled1 = model(inputs_unlabeled, 0)
                        outputs_unlabeled2 = model(inputs_unlabeled, 0)
                        cons_loss = torch.abs(outputs_unlabeled1 - outputs_unlabeled2).sum()
                        cons_loss = opt.beta * cons_loss / opt.batchSize
                        running_cons_loss += cons_loss.item() * opt.batchSize
                        loss += cons_loss
                        
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_cons_loss = running_cons_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if opt.sendNeptune:
                neptune.send_metric('{}_loss'.format(phase), epoch_loss)
                neptune.send_metric('{}_acc'.format(phase), epoch_acc)
                neptune.send_metric('{}_cons_loss'.format(phase), epoch_cons_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.model_ft.state_dict())

        print()
    torch.save(best_model_wts, "./{}.pth".format(opt.sessionName))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} achieved on model from epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    return model


def getConfig():
    # naprawic to tak, zeby nie byly potrzebne dwa parsery.
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--dataset', required=True, help='Cifar | ImageNet')
    parser.add_argument('--batchSize', type=int, required = True)
    parser.add_argument('--sessionName', required = True)
    parser.add_argument('--net', )
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weightDecay', type=float)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpuId', type=int, required=True)
    parser.add_argument('--beta', type=float)

    parserShell = argparse.ArgumentParser()
    parserShell.add_argument('--sendNeptune', action='store_true')
    parserShell.add_argument('--pretrained', action='store_true')
    parserShell.add_argument('--onlyLabeled', action='store_true')

    opt = parser.parse_args(['@config.txt'])
    optShell = parserShell.parse_args()

    opt.sendNeptune = optShell.sendNeptune
    opt.pretrained = optShell.pretrained
    opt.onlyLabeled = optShell.onlyLabeled
    opt.device = torch.device("cuda:{}".format(opt.gpuId) if torch.cuda.is_available() else "cpu")
    if opt.dataset == "Cifar":
        opt.outSize = 10
    else:
        opt.outSize = 100
    return opt
