import torch
import argparse

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


def getConfig():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--dataset', required=True, help='Cifar | ImageNet')
    parser.add_argument('--batchSize', type=int, required = True)
    parser.add_argument('--sessionName', required = True)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--net', )
    parser.add_argument('--sendNeptune', action='store_true')
    parser.add_argument('--cifarFactor', type=float, required = True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weightDecay', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpuId', type=int, required=True)
    

    opt = parser.parse_args(['@config.txt'])
    if opt.dataset == "Cifar":
        opt.outSize = 10
    else:
        opt.outSize = 100
    return opt
