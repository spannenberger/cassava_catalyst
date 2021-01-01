import torch
from torch import nn
import torchvision as vision
from catalyst.dl import registry
@registry.Model
class ResNextFacebook32x8d(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

        classifier_name, old_classifier = self.backbone._modules.popitem()
        if isinstance(old_classifier, nn.Sequential):
            input_shape = old_classifier[-1].in_features
            old_classifier[-1] = nn.Linear(input_shape,num_classes)

        elif isinstance(old_classifier, nn.Linear):
            input_shape = old_classifier.in_features
            old_classifier = nn.Linear(input_shape,num_classes)

        else:
            raise Exception("Uknown type of classifier {}".format(type(old_classifier)))
        self.backbone.add_module(classifier_name, old_classifier)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in getattr(self.backbone, classifier_name).parameters():
            param.requires_grad = True

    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = ResNextFacebook32x8d()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output)
