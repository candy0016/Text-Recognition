import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import os
import math
import models


class Trainer(object):
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.isTrain = opt.isTrain
        self.opt = opt
        self.metric = 0

        self.net = models.Res18(opt).to(opt.device)
        self.create_hook()
        
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = _get_scheduler(self.optimizer, opt)
            self.criterionBCE = nn.BCELoss().to(opt.device)

    def create_hook(self):
        '''Create the method to extract middle layer output.
        '''
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        
        self.net.net.layer1.register_forward_hook(get_activation('layer1'))
        self.net.net.layer2.register_forward_hook(get_activation('layer2'))
        self.net.net.layer3.register_forward_hook(get_activation('layer3'))
        self.net.net.layer4.register_forward_hook(get_activation('layer4'))

    def get_hook(self):
        '''
        Return middle layer output.
        Call this function after calling run().
        '''
        return self.activation['layer1'], \
                self.activation['layer2'], \
                self.activation['layer3'], \
                self.activation['layer4']

    def eval(self):
        self.net.eval()

    def run(self, data, pos_matrix=None):
        '''
        Input parameters:
            anc:        anchor image [batch, c(1), h, w]
            label:      label of anchor image [batch, num_class]
            label_num:  lael number of each image [batch, 1]
            pos_matrix: matrix of one random positive anchor of each classes [num_class, c(1), h, w]
        '''
        if self.opt.isTrain:
            self.optimizer.zero_grad()
            anc, label = data
            self.anc = anc.to(self.opt.device)
            self.label = label.to(self.opt.device)
            self.predict = self.net(self.anc)
        else:
            predict = self.net(data)
            return predict

    def compute_loss(self):
        self.loss_classify = self.criterionBCE(self.predict, self.label)
        self.loss_classify.backward()
        self.optimizer.step()

    def get_loss(self):
        """Return losses of current step.
        """
        dic = {}
        dic['loss_classify'] = self.loss_classify.item()
        return dic

    def save(self, epoch, best=False):
        save_filename = 'net_%s.pth' % (epoch)
        if best:
            save_filename = 'net_best.pth'
        save_path = os.path.join(self.opt.save_dir_model, save_filename)

        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.cpu().state_dict(), save_path)
            self.net.to(self.opt.device)
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def load(self):
        load_filename = 'net_%s.pth' % (self.opt.load_epoch)
        load_path = os.path.join(self.opt.weight_dir, load_filename)
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.opt.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.net, key.split('.'))
        self.net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.opt.lr_policy == 'plateau':
            self.scheduler.step(self.metric)
        else:
            self.scheduler.step()

        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))


def _get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler