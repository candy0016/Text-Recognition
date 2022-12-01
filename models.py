import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as mod


class Siamese(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, opt):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Siamese, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        dim = opt.classes
        pred_dim = opt.pred_dim
        self.encoder = mod.resnet18(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()


class Cosine_Loss(nn.Module):
    def __init__(self):
        super(Cosine_Loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, outp):
        p1, p2, z1, z2 = outp
        loss = -(self.cos(p1, z2).mean() + self.cos(p2, z1).mean()) * 0.5
        return loss

