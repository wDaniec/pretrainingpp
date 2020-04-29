import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import neptune
import utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Network():
    def __init__(self, opt):
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
    def __init__(self, opt):
        if opt.dataset == "Cifar":
            final_datasets = utils.getCifar(opt)
        elif opt.dataset == "ImageNet":
            final_datasets = utils.getImageNet(opt)
        elif opt.dataset == "ImageNet+Cifar":
            labeled_images = utils.getImageNet(opt)
            unlabeled_images = utils.getCifar(opt)
            final_datasets = {x: utils.ConcatDataset(labeled_images[x], unlabeled_images[x]) for x in ['train', 'val']}

        self.dataloaders = {x: torch.utils.data.DataLoader(final_datasets[x], batch_size=opt.batchSize,
                                                    shuffle=True, num_workers=4) for x in ['train', 'val']}


        self.dataset_sizes = {x: len(final_datasets[x]) for x in ['train', 'val']}

        print('val length:', len(final_datasets['val']))
        print('train length:', len(final_datasets['train']))


def run_train(opt):
    print(opt)

    if opt.sendNeptune:
        neptune.init('andrzejzdobywca/pretrainingpp')
        exp = neptune.create_experiment(name=opt.sessionName, params=vars(opt), tags=["best_experiment"]) 
    dataset = Dataset(opt)
    net = Network(opt)

    utils.train_model(opt, net, dataset)

    if opt.sendNeptune:
        neptune.stop()
        # exit(1)
    
if __name__=='__main__':
    opt = utils.getConfig()
    print(vars(opt))
    run_train(opt)