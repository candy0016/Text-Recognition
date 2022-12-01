import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import os
import math
import models
import torchvision.models as mod


class Trainer(object):
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.isTrain = opt.isTrain
        self.opt = opt

        self.net = mod.resnet18(num_classes=opt.classes, zero_init_residual=True).to(opt.device)
        
        if self.isTrain:
            self.init_lr = opt.lr * opt.batch_size / 256
            print('Initial learning rate: %.7f' % (self.init_lr))
            
            # Freeze all layers' weight but the last fc
            for name, param in self.net.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            # init the fc layer
            self.net.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.net.fc.bias.data.zero_()

            optim_params = list(filter(lambda p: p.requires_grad, self.net.parameters()))
            self.optimizer = torch.optim.SGD(optim_params, self.init_lr, momentum=opt.momentum, weight_decay=opt.wd)
            self.criterion = nn.BCELoss().to(opt.device)

    def eval(self):
        self.net.eval()

    def run(self, data):
        '''
        data (tuple):
            img:   anchor image [batch, c(1), h, w]
            label: label of anchor image [batch, num_class]
        '''
        if self.opt.isTrain:
            self.optimizer.zero_grad()
            img = data[0].to(self.opt.device)
            self.label = data[1].to(self.opt.device)
            self.predict = torch.sigmoid(self.net(img))
        else:
            predict = self.net(data)
            return torch.sigmoid(predict)

    def compute_loss(self):
        self.loss = self.criterion(self.predict, self.label)
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
        state_dict = torch.load(load_path, map_location=str(self.opt.device))
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('encoder') and not k.startswith('encoder.fc'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = self.net.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # Eval the model to prevent from BatchNorm gradient
        self.net.eval()
    
    def resume(self):
        load_filename = 'net_%s.pth' % (self.opt.load_epoch)
        load_path = os.path.join(self.opt.weight_dir, load_filename)
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.opt.device))
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def update_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * epoch / self.opt.n_epochs))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = self.init_lr
            else:
                param_group['lr'] = cur_lr
                print('encoder learning rate: %.7f' % (cur_lr))
