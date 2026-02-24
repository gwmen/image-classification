import torch
import timm


def create_model(model_name: str, pretrained=True):
    # IBN-Net series
    if model_name in ['resnet18_ibn_a', 'resnet34_ibn_a', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a']:
        model = torch.hub.load('XingangPan/IBN-Net', 'resnet18_ibn_a', pretrained=pretrained)
    else:
        models = timm.list_models()
        if model_name in models:
            model = timm.create_model(model_name, pretrained=pretrained)
        else:
            raise KeyError(f"model name [{model_name}] not in timm model zoo")
    return model
