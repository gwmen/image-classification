import torch
import timm

clip_info = {
    'resnet18': (-2, 512),
    'resnet34': (-2, 512),
    'resnet50': (-2, 2048),
    'resnet101': (-2, 2048),
    'resnet151': (-2, 2048),
}


def create_model(model_name: str, pretrained=True, clip_id=None, in_planes=None):
    # IBN-Net series
    if model_name in ['resnet18_ibn_a', 'resnet34_ibn_a', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a']:
        model = torch.hub.load('XingangPan/IBN-Net', 'resnet18_ibn_a', pretrained=True)
    else:
        models = timm.list_models()
        if model_name in models:
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise KeyError(f"model name [{model_name}] not in timm model zoo")
    clip_id = clip_id if clip_id else clip_info[model_name][0]
    clip_plane = in_planes if in_planes else clip_info[model_name][1]
    # resnet 5,6,7,8
    # return torch.nn.Sequential(*list(model.children())[:clip_id]), clip_plane
    _model = [o for o in torch.nn.Sequential(*list(model.children()))]
    return [torch.nn.Sequential(*_model[:5]),
            torch.nn.Sequential(_model[5]),
            torch.nn.Sequential(_model[6]),
            torch.nn.Sequential(_model[7])], clip_plane

# print()
# model1 = timm.create_model(
#     "seresnext50_32x4d",
#     pretrained=True,
# )

# # 查看所有 SE-ResNeXt 相关模型
# all_mm = []
# # print("可用的 SE-ResNeXt 模型:")

# import torch
# import torchvision.models
# model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
# for name in timm.list_models('*bn*'):
#     print(name)

# if 'ibn' in name.lower():
#     print(f"  - {name}")
#         all_mm.append(name)
# print(all_mm)
# # ['legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext26ts', 'seresnext50_32x4d', 'seresnext101_32x4d', 'seresnext101_32x8d', 'seresnext101_64x4d', 'seresnext101d_32x8d', 'seresnextaa101d_32x8d', 'seresnextaa201d_32x8d']
#
# #  convnext_nano
# #   - convnext_small
# #   - convnext_tiny
# # convnext_large
