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
        self.init_lr = opt.lr * opt.batch_size / 256
        print('Initial learning rate: %.7f' % (self.init_lr))

        self.net = models.Siamese(opt).to(opt.device)
        
        if self.isTrain:
            optim_params = [{'params': self.net.encoder.parameters(), 'fix_lr': False},
                            {'params': self.net.predictor.parameters(), 'fix_lr': True}]
            self.optimizer = torch.optim.SGD(optim_params, self.init_lr, momentum=opt.momentum, weight_decay=opt.wd)
            self.criterionCOS = models.Cosine_Loss().to(opt.device)

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
            img1 = data[0].to(self.opt.device)
            img2 = data[1].to(self.opt.device)
            self.predict = self.net(img1, img2)
        else:
            predict = self.net(data)
            return predict

    def compute_loss(self):
        self.loss = self.criterionCOS(self.predict)
        self.loss.backward()
        self.optimizer.step()

    def get_loss(self):
        """Return losses of current step.
        """
        dic = {}
        dic['loss'] = self.loss.item()
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

    def update_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * epoch / self.opt.n_epochs))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = self.init_lr
            else:
                param_group['lr'] = cur_lr
                print('encoder learning rate: %.7f' % (cur_lr))
        
