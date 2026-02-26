# encoding: utf-8
"""
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from .backbones.models import create_model
from .plugin.pim_module import PluginModel


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


NODES = {
    'resnet50':
        {
            'layer1.2.act3': 'layer1',
            'layer2.3.act3': 'layer2',
            'layer3.5.act3': 'layer3',
            'layer4.2.act3': 'layer4',
        },
    'resnet18':
        {
            'layer1.1.act2': 'layer1',
            'layer2.1.act2': 'layer2',
            'layer3.1.act2': 'layer3',
            'layer4.1.act2': 'layer4',
        },
    'resnet34':
        {
            'layer1.2.act2': 'layer1',
            'layer2.3.act2': 'layer2',
            'layer3.5.act2': 'layer3',
            'layer4.2.act2': 'layer4',
        }
}


class Baseline(nn.Module):
    medium_num = 1024

    def __init__(self, num_classes, num_domain, last_stride, model_path, neck, clip_id, in_planes, neck_feat,
                 model_name, cfg):
        super(Baseline, self).__init__()
        backbone = create_model(model_name)
        return_nodes = NODES[model_name]
        self.in_planes = backbone.fc.in_features
        self.feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.shell_extractor = None
        if cfg.PLUG_MODEL.ENABLE:
            num_selects = dict()
            [num_selects.update({layer_name: select_size}) for layer_name, select_size
             in zip(cfg.PLUG_MODEL.NUM_SELECTS_NAME, cfg.PLUG_MODEL.NUM_SELECTS_SIZE)]
            self.shell_extractor = PluginModel(backbone=self.feature_extractor,
                                               return_nodes=return_nodes,
                                               img_size=cfg.INPUT.SIZE_TRAIN[0],
                                               use_fpn=cfg.PLUG_MODEL.USE_FPN,
                                               fpn_size=cfg.PLUG_MODEL.FPN_SIZE,
                                               proj_type=cfg.PLUG_MODEL.PROJ_TYPE,
                                               upsample_type=cfg.PLUG_MODEL.UP_SAMPLE_TYPE,
                                               use_selection=True,
                                               num_classes=num_classes,
                                               num_selects=num_selects,
                                               use_combiner=True,
                                               comb_proj_size=None)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.Linear(self.in_planes, self.medium_num),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(self.medium_num, self.num_classes)
            nn.Linear(self.in_planes, self.num_classes)
        )
        # self.neck = neck
        # self.neck_feat = neck_feat
        #
        # if self.neck == 'no':
        #     self.classifier = nn.Sequential(
        #         nn.Dropout(0.5),
        #         nn.Linear(self.in_planes, self.medium_num),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.3),
        #         nn.Linear(self.medium_num, self.num_classes)
        #     )
        #     # self.classifier = nn.Linear(self.in_planes, self.num_classes)
        #     # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
        #     # self.classifier.apply(weights_init_classifier)  # new add by luo
        # elif self.neck == 'bnneck':
        #     self.bottleneck = nn.BatchNorm1d(self.in_planes)
        #     self.bottleneck.bias.requires_grad_(False)  # no shift
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #
        #     self.bottleneck.apply(weights_init_kaiming)
        #     self.classifier.apply(weights_init_classifier)
        # self.adapter = nn.Sequential(
        #     nn.Conv2d(1024, 384, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(384)
        # )

    #             self.projection = nn.Sequential(
    #             nn.Linear(resnet_feat_dim, hidden_dim),
    #             nn.GELU(),
    #             nn.Linear(hidden_dim, hidden_dim)
    #         )

    def forward(self, x):

        if self.shell_extractor:
            out = self.shell_extractor(x)
        else:
            extract_features = self.feature_extractor(x)

            last_fea = extract_features['layer4']
            gap_fea = self.gap(last_fea)  # (b, 2048, 1, 1)
            reshape_fea = gap_fea.view(gap_fea.shape[0], -1)  # flatten to (bs, 2048)

            cls_score = self.classifier(reshape_fea)
            out = {'comb_outs': cls_score}
        return out, None  # student feature for distill

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
