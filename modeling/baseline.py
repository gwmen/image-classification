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
from .plugin.common_module import ClassificationModel
from .plugin.fpn_module import FPNClassificationModel

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

    def __init__(self, num_classes, num_domain, last_stride, model_path, neck, clip_id, in_planes, neck_feat,
                 model_name, cfg):
        super(Baseline, self).__init__()
        backbone = create_model(model_name)
        return_nodes = NODES[model_name]
        in_planes = backbone.fc.in_features
        feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        if cfg.PLUG_MODEL.ENABLE:
            num_selects = dict()
            [num_selects.update({layer_name: select_size}) for layer_name, select_size
             in zip(cfg.PLUG_MODEL.NUM_SELECTS_NAME, cfg.PLUG_MODEL.NUM_SELECTS_SIZE)]
            self.shell_extractor = PluginModel(feature_extractor,
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
        else:
            if False:
                self.shell_extractor = ClassificationModel(feature_extractor, in_planes=in_planes,
                                                           num_classes=num_classes)
            else:
                num_selects = dict()
                [num_selects.update({layer_name: select_size}) for layer_name, select_size
                 in zip(cfg.PLUG_MODEL.NUM_SELECTS_NAME, cfg.PLUG_MODEL.NUM_SELECTS_SIZE)]
                self.shell_extractor = FPNClassificationModel(feature_extractor,
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
                                                              comb_proj_size=None,
                                                              in_planes=in_planes
                                                              )

    def forward(self, x):
        out = self.shell_extractor(x)
        return out, None  # student feature for distill

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
