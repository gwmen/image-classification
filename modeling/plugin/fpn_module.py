import torch
import torch.nn as nn
from typing import Union
import copy


class FPN(nn.Module):

    def __init__(self, inputs: dict, fpn_size: int, proj_type: str, upsample_type: str):
        """
        inputs : dictionary contains torch.Tensor
                 which comes from backbone output
        fpn_size: integer, fpn
        proj_type:
            in ["Conv", "Linear"]
        upsample_type:
            in ["Bilinear", "Conv", "Fc"]
            for convolution neural network (e.g. ResNet, EfficientNet), recommand 'Bilinear'.
            for Vit, "Fc". and Swin-T, "Conv"
        """
        super(FPN, self).__init__()
        assert proj_type in ["Conv", "Linear"], \
            "FPN projection type {} were not support yet, please choose type 'Conv' or 'Linear'".format(proj_type)
        assert upsample_type in ["Bilinear", "Conv"], \
            "FPN upsample type {} were not support yet, please choose type 'Bilinear' or 'Conv'".format(proj_type)

        self.fpn_size = fpn_size
        self.upsample_type = upsample_type
        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            if proj_type == "Conv":
                m = nn.Sequential(
                    nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),
                    nn.ReLU(),
                    nn.Conv2d(inputs[node_name].size(1), fpn_size, 1)
                )
            elif proj_type == "Linear":
                m = nn.Sequential(
                    nn.Linear(inputs[node_name].size(-1), inputs[node_name].size(-1)),
                    nn.ReLU(),
                    nn.Linear(inputs[node_name].size(-1), fpn_size),
                )
            self.add_module("Proj_" + node_name, m)

            ### upsample module
            if upsample_type == "Conv" and i != 0:
                assert len(inputs[node_name].size()) == 3  # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i - 1]].size(1)
                if in_dim != out_dim:
                    m = nn.Conv1d(in_dim, out_dim, 1)  # for spatial domain
                else:
                    m = nn.Identity()
                self.add_module("Up_" + node_name, m)

        if upsample_type == "Bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def upsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x1_name: str):
        """
        return Upsample(x1) + x1
        """
        if self.upsample_type == "Bilinear":
            if x1.size(-1) != x0.size(-1):
                x1 = self.upsample(x1)
        else:
            x1 = getattr(self, "Up_" + x1_name)(x1)
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        hs = []
        prj_out = dict()
        for i, name in enumerate(x):
            # x[name] = getattr(self, "Proj_" + name)(x[name])
            prj_out[name] = getattr(self, "Proj_" + name)(x[name])
            hs.append(name)

        for i in range(len(hs) - 1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i - 1]
            prj_out[x0_name] = self.upsample_add(prj_out[x0_name],
                                                 prj_out[x1_name],
                                                 x1_name)
        return prj_out


class FPNClassificationModel(nn.Module):

    def __init__(self,
                 backbone: torch.nn.Module,
                 return_nodes: Union[dict, None],
                 img_size: int,
                 use_fpn: bool,
                 fpn_size: Union[int, None],
                 proj_type: str,
                 upsample_type: str,
                 use_selection: bool,
                 num_classes: int,
                 num_selects: dict,
                 use_combiner: bool,
                 comb_proj_size: Union[int, None],
                 in_planes=2048
                 ):

        super(FPNClassificationModel, self).__init__()

        self.backbone = backbone

        rand_in = torch.randn(1, 3, img_size, img_size)
        outs = self.backbone(rand_in)

        self.fpn = FPN(outs, fpn_size, proj_type, upsample_type)
        self.build_fpn_classifier(outs, fpn_size, num_classes)
        self.fpn_size = fpn_size
        self.fc = nn.Linear(sum([(img_size // o) ** 2 for o in [4, 8, 16, 32]]), 1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(in_planes, num_classes)

    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                nn.Conv1d(fpn_size, fpn_size, 1),
                nn.BatchNorm1d(fpn_size),
                nn.ReLU(),
                nn.Conv1d(fpn_size, num_classes, 1)
            )
            self.add_module("fpn_classifier_" + name, m)

    def forward_backbone(self, x):
        return self.backbone(x)

    def fpn_predict(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            # predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H * W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            logits[name] = getattr(self, "fpn_classifier_" + name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous()  # transpose

    def forward(self, x: torch.Tensor):

        logits = {}
        x = self.forward_backbone(x)
        fpn_fea = self.fpn(x)
        self.fpn_predict(fpn_fea, logits)
        tensors = list()
        for name in logits:
            tensors.append(logits[name])
        cat_tensor = torch.cat(tensors, dim=1).transpose(1, 2).contiguous()  # B, S', C --> B, C, S
        pool_tensor = self.fc(cat_tensor).view(cat_tensor.shape[0], -1)
        return {'comb_outs': pool_tensor}
