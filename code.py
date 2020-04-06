import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import neptune
import sys
import utils


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)

opt = utils.getConfig()
print(opt)

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
            labeled_images = utils.getCifar(opt)
        elif opt.dataset == "ImageNet":
            labeled_images = utils.getImageNet(opt)
            unlabeled_images = utils.getCifar(opt)['train']
            if not opt.onlyLabeled:
                unlabeled_images = utils.getCifarUnlabeled(opt)
                # tak naprawde to jest labeled, ale bede ignorowal
                self.dataloader_unlabeled = torch.utils.data.DataLoader(unlabeled_images, batch_size=opt.batchSize, 
                                                    shuffle=True, num_workers=4)

        self.dataloaders = {x: torch.utils.data.DataLoader(labeled_images[x], batch_size=opt.batchSize,
                                                    shuffle=True, num_workers=4) for x in ['train', 'val']}


        self.dataset_sizes = {x: len(labeled_images[x]) for x in ['train', 'val']}

        print('val length:', len(labeled_images['val']))
        print('train length:', len(labeled_images['train']))


dataset = Dataset()
net = Network()

utils.train_model(opt, net, dataset)

if opt.sendNeptune:
    neptune.stop()
    exit(1)