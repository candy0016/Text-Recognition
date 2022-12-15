import torch.nn as nn
import torch


def Resnet18(opt):
    return Resnet(BasicBlock, [2, 2, 2, 2], opt.classes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.final = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        _x = self.avgpool(x)
        x = torch.flatten(_x, 1)
        x = self.fc(x)
        x = self.final(x)
        return x, _x


def get_triplet_pairs(indices):
    """Return index numbers of each triplet.
    """
    cand = indices
    pair_list = []
    for i in range(len(cand)):
        lab = cand[i]
        same_class_idx = torch.nonzero(cand==lab, as_tuple=False).reshape(-1) # [num_objs]
        diff_class_idx = torch.nonzero(cand!=lab, as_tuple=False).reshape(-1)
        pos = same_class_idx[same_class_idx>i]
        neg = diff_class_idx[diff_class_idx>i]
        if pos.shape[0]>0 and neg.shape[0]>0:
            dic = {'anc': i, 'pos': pos, 'neg': neg}
            pair_list.append(dic)
    return pair_list


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.dis = nn.PairwiseDistance(p=2)
        self.cos = nn.CosineSimilarity(dim=0)
        self.margin = 1.0

    def forward(self, embs, indices):
        """
        embs:    [batch*2, feature_shape] middle layer output of each input
        indices: [batch*2, 1] a list of index of label for embs
        """
        loss_all = []
        pairs = get_triplet_pairs(indices)
        for p in pairs:
            anc = embs[p['anc']]
            for pos in p['pos']:
                positive_dist = 1. - self.cos(anc, embs[pos])
                for neg in p['neg']:
                    negative_dist = 1. - self.cos(anc, embs[neg])
                    loss = torch.clamp(positive_dist - negative_dist + self.margin, min=0.0)
                    loss_all.append(loss)
        outp = torch.sum(torch.stack(loss_all))
        return outp