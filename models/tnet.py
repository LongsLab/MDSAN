import scipy.misc
import scipy.io
from ops import *
import math
import torch
from torchvision import models
from torch import nn
import clip
from config import opt

def init_parameters_recursively(layer):
    if isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            init_parameters_recursively(sub_layer)
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, std=0.01)
    else:
        return

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.cnn = models.vgg19(pretrained=True).type(torch.float32).features
        self.feature = nn.Sequential(
            nn.Linear(512 * 7 * 7, SEMANTIC_EMBED * 2),
        )
        self.model, self.preprocess = clip.load('ViT-B/32', 'cuda')
        self.feature1=nn.Sequential(
            nn.Linear(512 , SEMANTIC_EMBED * 2),
        )
        self.hash = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, opt.bit),
            nn.Tanh()
        )
        self.label = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, opt.label_dim),
            nn.Sigmoid()
        )
        self.cross_feature = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, SEMANTIC_EMBED),
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.label)
        init_parameters_recursively(self.cross_feature)

    def forward(self, inputs):
        base = self.cnn(inputs).view(inputs.shape[0], -1)
        base = base.type(torch.float32)
        Vit =self.model.encode_image(inputs).view(inputs.shape[0], -1)
        Vit = Vit.type(torch.float32)

        feat_I = torch.cat((self.feature(base), self.feature1(Vit)), dim=1)
        fea_T_pred = self.cross_feature(fea_I)
        hsh_I = self.hash(fea_I)
        lab_I = self.label(fea_I)
        return torch.squeeze(fea_I), torch.squeeze(hsh_I), torch.squeeze(lab_I)

class LabelNet(nn.Module):
    def __init__(self):
        super(LabelNet, self).__init__()
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=opt.bit, kernel_size=(1, opt.label_dim), stride=(1, 1), bias=False),
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.hash)

    def forward(self, inputs):
        hsh_I = self.hash(inputs.view(inputs.shape[0], 1, 1, -1))
        return torch.squeeze(hsh_I)

class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()
        self.norm = nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0)
        self.hash = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, opt.bit),
            nn.Tanh()
        )
        self.label = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, opt.label_dim),
            nn.Sigmoid()
        )
        self.init_parameters()
        self.gconv1 = nn.Linear(opt.y_dim, 2048)
        self.BN1 = nn.BatchNorm1d(2048)
        self.act1 = nn.ReLU()
        self.gconv2 = nn.Linear(2048, 2048)
        self.BN2 = nn.BatchNorm1d(2048)
        self.act2 = nn.ReLU()
        self.gconv3 = nn.Linear(2048, 512)
        self.alpha = 1.0
    def init_parameters(self):
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.label)
    def forward(self, inputs , in_affnty, out_affnty):
        out = self.gconv1(torch.squeeze(inputs))  # inputs:32,1386
        out = in_affnty.mm(out)
        out = self.BN1(out)
        out = self.act1(out)
        # block 2
        out = self.gconv2(out)
        out = out_affnty.mm(out)
        out = self.BN2(out)
        out = self.act2(out)
        # block 3
        out = self.gconv3(out)
        out = torch.tanh(self.alpha * out)
        hash_gcn=self.hash(out)
        label=self.label(out)

        return torch.squeeze(out), torch.squeeze(hash_gcn), torch.squeeze(label)

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

