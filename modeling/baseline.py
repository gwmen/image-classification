# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

# from .backbones.resnet import ResNet, BasicBlock, Bottleneck
# from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
# from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.models import create_model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):

    def __init__(self, num_classes, num_domain, last_stride, model_path, neck, clip_id, in_planes, neck_feat,
                 model_name,
                 pretrain_choice):
        super(Baseline, self).__init__()
        # 18,34  -- 512
        # resnet50 -2048
        # self.base, self.in_planes = create_model(model_name, True, clip_id, in_planes)
        [self.p1, self.p2, self.p3, self.p4], self.in_planes = create_model(model_name, True, clip_id, in_planes)
        self.base = [self.p1, self.p2, self.p3, self.p4]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        self.adapter = nn.Sequential(
            nn.Conv2d(1024, 384, kernel_size=1, stride=1),
            nn.BatchNorm2d(384)
        )

    #             self.projection = nn.Sequential(
    #             nn.Linear(resnet_feat_dim, hidden_dim),
    #             nn.GELU(),
    #             nn.Linear(hidden_dim, hidden_dim)
    #         )

    def forward(self, x):
        # base_fea = self.base(x)
        base_fea = []
        for _m in self.base:
            x = _m(x)
            base_fea.append(x)

        stu_feat = self.adapter(base_fea[-2])
        global_feat = self.gap(base_fea[-1])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        # if self.neck == 'no':
        #     feat = global_feat
        # elif self.neck == 'bnneck':
        #     feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # if self.training:
        #     cls_score = self.classifier(feat)
        #     return cls_score, global_feat  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat
        cls_score = self.classifier(global_feat)
        return cls_score, stu_feat  # student feature for distill

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
