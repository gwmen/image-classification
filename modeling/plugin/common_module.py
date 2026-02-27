import torch
from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, backbone: torch.nn.Module, in_planes: int, num_classes: int):
        super(ClassificationModel, self).__init__()
        self.feature_extractor = backbone
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.in_planes = in_planes
        self.classifier = nn.Sequential(
            nn.Linear(self.in_planes, self.num_classes)
        )

    def forward(self, x):
        extract_features = self.feature_extractor(x)

        last_fea = extract_features['layer4']
        gap_fea = self.gap(last_fea)  # (b, 2048, 1, 1)
        reshape_fea = gap_fea.view(gap_fea.shape[0], -1)  # flatten to (bs, 2048)

        cls_score = self.classifier(reshape_fea)
        out = {'comb_outs': cls_score}
        return out

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
