import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import neptune
import sys
import utils


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)

opt = utils.getConfig()

if opt.sendNeptune:
    neptune.init('andrzejzdobywca/pretrainingpp')
    exp = neptune.create_experiment(name=opt.sessionName)
    exp.log_artifact('config.txt')

class Network():
    def __init__(self):
        model_ft = models.mobilenet_v2()
        num_ftrs = model_ft.classifier[1].in_features

        model_ft.classifier[1] = nn.Linear(num_ftrs, 100)
        if opt.pretrained:
            model_ft.load_state_dict(torch.load(opt.net))
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, opt.outSize)

        self.model = model_ft.to(opt.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model_ft.parameters(), weight_decay=opt.weightDecay, lr=opt.lr)



class Dataset():
    def __init__(self):
        if opt.dataset == "Cifar":
            image_datasets = {'train': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=True, download=True, transform=opt.transform), 
                        'val': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=False, download=True, transform=opt.transform)}
            train_ds, _ = utils.trainTestSplit(image_datasets['train'], opt.cifarFactor)
            image_datasets['train'] = train_ds
        else:
            image_datasets = {x: datasets.ImageFolder(os.path.join('~/data/lilImageNet', x),
                                                transform=opt.transform)
                        for x in ['train', 'val']}
        

        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchSize,
                                                    shuffle=True, num_workers=4) for x in ['train', 'val']}
        
        self.dataset_sizes = dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        print('val length:', len(image_datasets['val']))
        print('train length:', len(image_datasets['train']))


dataset = Dataset()
net = Network()

utils.train_model(opt, net, dataset)

if opt.sendNeptune:
    neptune.stop()
    exit(1)