# encoding: utf-8
"""
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
# from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from .backbones.models import create_model
from .plugin.base_module import BaseClassification

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

HEADERS = {
    'base': BaseClassification,
}


class Baseline(nn.Module):

    def __init__(self, model_name, head, num_classes, arguments=None):
        super(Baseline, self).__init__()
        backbone = create_model(model_name)
        return_nodes = NODES[model_name]
        in_planes = backbone.fc.in_features
        self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.header = HEADERS[head](in_planes=in_planes, num_classes=num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        out = self.header(feature)
        return out, None

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
