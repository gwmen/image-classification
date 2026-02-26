from utils.yaml_reader import load_yaml
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names
from modeling.plugin.pim_module import PluginModel

cfg = load_yaml(r'e:\20260130\FGVC Good job\image-classification\configs\pim_config.yml')


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


model = Model()

print(model)  ### structure
print(get_graph_node_names(model))
# if we want conv1 output and conv2 output
return_nodes = {
    "conv1.5": "layer1",
    "conv2.5": "layer2",
}
# notice that 'layer1' and 'layer2' must match return_nodes's value
num_selects = {
    "layer1": 64,
    "layer2": 64
}

IMG_SIZE = 224
USE_FPN = True
FPN_SIZE = 128  # fpn projection size, if do not use fpn, you can set fpn_size to None

PROJ_TYPE = "Conv"
UPSAMPLE_TYPE = "Bilinear"

pim_model = \
    PluginModel(backbone=model,
                return_nodes=return_nodes,
                img_size=IMG_SIZE,
                use_fpn=USE_FPN,
                fpn_size=FPN_SIZE,
                proj_type=PROJ_TYPE,
                upsample_type=UPSAMPLE_TYPE,
                use_selection=True,
                num_classes=10,
                num_selects=num_selects,
                use_combiner=True,
                comb_proj_size=None)

rand_inp = torch.randn(1, 3, 224, 224)
outs = pim_model(rand_inp)

print([name for name in outs])

print()
