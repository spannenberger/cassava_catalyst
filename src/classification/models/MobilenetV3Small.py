import torch
from torch import nn
import torchvision as vision
from models.ClassifierReplacer import replace_classifier

class MobilenetV3Small(nn.Module):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        super().__init__()
        self.path = path
        self.diff_classes_flag = diff_classes_flag
        self.is_local = is_local
        if self.is_local:
            if self.diff_classes_flag:
                self.backbone = vision.models.mobilenet_v3_small(pretrained=False)
                replace_classifier(self.backbone, old_num_classes)
                self.load_state_dict(torch.load(self.path)['model_state_dict'])
                replace_classifier(self.backbone, num_classes)
            else:
                self.backbone = vision.models.mobilenet_v3_small(pretrained=False)
                replace_classifier(self.backbone, num_classes)
                self.load_state_dict(torch.load(self.path)['model_state_dict'])
        else:
            self.backbone = vision.models.mobilenet_v3_small(pretrained=True)
            replace_classifier(self.backbone, num_classes)
    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = MobilenetV3Small()
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)

    print(output)
