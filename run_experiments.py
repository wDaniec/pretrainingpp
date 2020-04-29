from code import run_train
import utils
import copy

def get_clean(opt, hparams):
    # moze ten deep copy wcale nie jest potrzebny
    lr, weightDecay, beta = hparams
    opt = copy.deepcopy(opt)
    opt.dataset = "ImageNet"
    opt.epochs = 40
    opt.batchSize = 128
    opt.lr = lr
    opt.weightDecay = weightDecay
    opt.beta = beta
    opt.sessionName = "ImageNet"
    opt.onlyLabeled = True
    return opt

def get_ssl(opt, hparams):
    lr, weightDecay, beta = hparams
    opt = copy.deepcopy(opt)
    opt.dataset = "ImageNet"
    opt.epochs = 40
    opt.batchSize = 128
    opt.lr = lr
    opt.weightDecay = weightDecay
    opt.beta = beta
    opt.sessionName = "ImageNet+Cifar"
    return opt

def get_pretrained_clean(opt, hparams):
    lr ,weightDecay, beta = hparams
    opt = copy.deepcopy(opt)
    opt.dataset = "Cifar"
    opt.epochs = 80
    opt.batchSize = 128
    opt.lr = lr
    opt.weightDecay = weightDecay
    opt.beta = beta
    opt.onlyLabeled = True
    opt.pretrained = True
    return opt

def get_pretrained_ssl(opt, hparams):
    lr, weightDecay, beta = hparams
    opt = copy.deepcopy(opt)
    opt.dataset = "Cifar"
    opt.epochs = 80
    opt.batchSize = 128
    opt.lr = lr
    opt.weightDecay = weightDecay
    opt.beta = beta
    opt.onlyLabeled = True
    opt.pretrained = True
    return opt



def run_multiple_with_pretrain():
    opt = utils.getConfig()
    hparams = (0.0005, 0.00004, 0.005)
    opt_clean = get_clean(opt, hparams)
    opt_ssl = get_ssl(opt, hparams)
    opt_pretrained_clean = get_pretrained_clean(opt, hparams)
    opt_pretrained_ssl = get_pretrained_ssl(opt, hparams)
    sessionName = "ImageNet"
    for i in range(4,6):
        opt_clean.sessionName = "{}_id:{}".format(sessionName, i)
        run_train(opt_clean)

        opt_ssl.sessionName = "{}+Cifar_id:{}".format(sessionName, i)
        run_train(opt_ssl)

        opt_pretrained_clean.sessionName = "{}->Cifar_id:{}".format(sessionName, i)
        opt_pretrained_clean.net = "./"+opt_clean.sessionName+".pth"
        run_train(opt_pretrained_clean)

        opt_pretrained_ssl.sessionName = "{}+Cifar->Cifar_id:{}".format(sessionName, i)
        opt_pretrained_ssl.net = "./"+opt_ssl.sessionName+".pth"
        run_train(opt_pretrained_ssl)



if __name__=='__main__':
    run_multiple_with_pretrain()