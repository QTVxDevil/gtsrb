import torch
import torch.nn as nn
from torchvision import models
from models.stn import SpatialTransformer

class ResNetWithSTN(nn.Module):
    def __init__(self, num_classes, stn_filters=(16, 32), stn_fc_units=128, input_size=(48, 48)):
        super(ResNetWithSTN, self).__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        try:
            self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet pretrained model: {e}")
        
        self.conv1_modified = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_modified = self.resnet.bn1
        self.relu_modified = self.resnet.relu
        self.maxpool_modified = nn.Identity()

        self.stn = SpatialTransformer(
            input_channels=64,
            filters_1=stn_filters[0],
            filters_2=stn_filters[1],
            fc_units=stn_fc_units,
            image_size=input_size[0]
        )

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.avgpool = self.resnet.avgpool
        num_features = self.resnet.fc.in_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

        nn.init.kaiming_normal_(self.conv1_modified.weight, mode='fan_out', nonlinearity='relu')

    def unfreeze_layers(self, layer_names):
        for name, module in self.named_modules():
            if name in layer_names:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfrozen layer: {name}")

    def forward(self, x):
        x = self.conv1_modified(x)
        x = self.bn1_modified(x)
        x = self.relu_modified(x)
        x = self.maxpool_modified(x)

        x = self.stn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x