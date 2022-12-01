import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as mod


class Res18(nn.Module):
    def __init__(self, opt):
        super(Res18, self).__init__()
        self.net = mod.resnet18(num_classes=opt.classes, zero_init_residual=True)
        self.final = nn.Sigmoid()

    def forward(self, x):
        y = self.net(x)
        y = self.final(y)
        return y


class Cosine_Loss(nn.Module):
    def __init__(self):
        super(Cosine_Loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, outp):
        p1, p2, z1, z2 = outp
        loss = -(self.cos(p1, z2).mean() + self.cos(p2, z1).mean()) * 0.5
        return loss


