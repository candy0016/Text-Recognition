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

        self.net = models.Resnet18(opt).to(opt.device)
        
        if self.isTrain:
            optim_params = self.net.parameters()
            self.optimizer = torch.optim.SGD(optim_params, self.init_lr, momentum=opt.momentum, weight_decay=opt.wd)
            self.criterionTL = models.TripletLoss().to(opt.device)
            self.criterionBCE = nn.BCELoss().to(opt.device)

    def eval(self):
        self.net.eval()

    def run(self, data):
        '''
        Input parameters:
            img:        blurred/clear image [batch, c(1), h, w]
            label:      label of anchor image [batch, num_class]
            label_idx:  index of label of each image [batch, 1]
        '''
        if self.opt.isTrain:
            self.optimizer.zero_grad()
            img1 = data[0].to(self.opt.device)
            img2 = data[1].to(self.opt.device)
            self.label = torch.cat((data[2], data[2]), 0).to(self.opt.device)
            self.label_idx = torch.cat((data[3], data[3]), 0).to(self.opt.device)

            y1, y1_emb = self.net(img1)
            y2, y2_emb = self.net(img2)
            self.ys = torch.cat((y1, y2), 0)
            self.embs = torch.cat((y1_emb, y2_emb), 0)
        else:
            predict, _ = self.net(data)
            return predict

    def compute_loss(self):
        self.loss_feature = self.criterionTL(self.embs, self.label_idx)
        self.loss_classify = self.criterionBCE(self.ys, self.label)
        self.loss = self.loss_feature*self.opt.weight_tl + self.loss_classify
        self.loss.backward()
        self.optimizer.step()

    def get_loss(self):
        """Return losses of current step.
        """
        dic = {}
        dic['loss_feature'] = self.loss_feature.item()
        dic['loss_classify'] = self.loss_classify.item()
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
        