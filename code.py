import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import neptune
import sys
import utils


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)

opt = utils.getConfig()
print(opt)

print(opt.sendNeptune)

if opt.sendNeptune:
    neptune.init('andrzejzdobywca/pretrainingpp')
    exp = neptune.create_experiment(name=opt.sessionName)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if opt.dataset == "Cifar":
    image_datasets = {'train': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=True, download=True, transform=transform), 
                'val': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=False, download=True, transform=transform)}
    train_ds, _ = utils.trainTestSplit(image_datasets['train'], opt.cifarFactor)
    image_datasets['train'] = train_ds
else:
    image_datasets = {x: datasets.ImageFolder(os.path.join('~/data/lilImageNet', x),
                                          transform=transform)
                  for x in ['train', 'val']}
 

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchSize,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print('val length:', len(image_datasets['val']))
print('train length:', len(image_datasets['train']))
device = torch.device("cuda:{}".format(opt.gpuId) if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    folder_name = "experiments/{}".format(opt.sessionName)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                inputs = inputs.to(device)
                labels = labels.to(device)

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
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if opt.sendNeptune:
                neptune.send_metric('{}_loss'.format(phase), epoch_loss)
                neptune.send_metric('{}_acc'.format(phase), epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "./{}/best_model.pth".format(opt.sessionName, epoch))
    return model


model_ft = models.mobilenet_v2()
num_ftrs = model_ft.classifier[1].in_features

model_ft.classifier[1] = nn.Linear(num_ftrs, 100)
if opt.pretrained:
    model_ft.load_state_dict(torch.load(opt.net))


num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier[1] = nn.Linear(num_ftrs, opt.outSize)

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), weight_decay=opt.weightDecay, lr=opt.lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

# exp_lr_scheduler jest wykomentowany!
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=180)
if SEND_NEPTUNE:
    neptune.stop()