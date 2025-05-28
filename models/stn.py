import torch
import torch.nn as nn
import torch.nn.functional as F

class Localization(nn.Module):
    def __init__(self, input_channels, filters_1, filters_2, fc_units, kernel_size=3, pool_size=2, image_size=(48, 48)):
        super(Localization, self).__init__()
        
        self.localization = nn.Sequential(
            
            nn.Conv2d(input_channels, filters_1, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(filters_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size, stride=pool_size), 

            nn.Conv2d(filters_1, filters_2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(filters_2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size, stride=pool_size), 
        )

        dummy_input_size = (1, input_channels, image_size, image_size)
        dummy_input = torch.zeros(dummy_input_size)
        with torch.no_grad():
            flattened_size = self.localization(dummy_input).view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, fc_units)
        self.fc2 = nn.Linear(fc_units, 6)

        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = self.localization(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        theta = self.fc2(x)
        theta = theta.view(-1, 2, 3)
        return theta

class SpatialTransformer(nn.Module):
    def __init__(self, input_channels, filters_1, filters_2, fc_units, image_size=48):
        super(SpatialTransformer, self).__init__()
        self.localization_net = Localization(input_channels, filters_1, filters_2, fc_units, image_size=image_size)

    def forward(self, x):
        theta = self.localization_net(x)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x