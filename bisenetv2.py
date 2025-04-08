import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class BiSeNet(nn.Module):
    def __init__(self, num_classes=256):
        super(BiSeNet, self).__init__()
        self.backbone = deeplabv3_resnet50(pretrained=False)  # No need for pretrained
        self.backbone.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Output layer

    def forward(self, x):
        return self.backbone(x)["out"]
