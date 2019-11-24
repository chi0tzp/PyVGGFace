import torch
import torch.nn as nn
from collections import OrderedDict


class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()

        self.features = nn.ModuleDict(OrderedDict(
            {
                # === Block 1 ===
                'conv_1_1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                'relu_1_1': nn.ReLU(inplace=True),
                'conv_1_2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                'relu_1_2': nn.ReLU(inplace=True),
                'maxp_1_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 2 ===
                'conv_2_1': nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                'relu_2_1': nn.ReLU(inplace=True),
                'conv_2_2': nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                'relu_2_2': nn.ReLU(inplace=True),
                'maxp_2_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 3 ===
                'conv_3_1': nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                'relu_3_1': nn.ReLU(inplace=True),
                'conv_3_2': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_2': nn.ReLU(inplace=True),
                'conv_3_3': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_3': nn.ReLU(inplace=True),
                'maxp_3_3': nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                # === Block 4 ===
                'conv_4_1': nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                'relu_4_1': nn.ReLU(inplace=True),
                'conv_4_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_2': nn.ReLU(inplace=True),
                'conv_4_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_3': nn.ReLU(inplace=True),
                'maxp_4_3': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 5 ===
                'conv_5_1': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_1': nn.ReLU(inplace=True),
                'conv_5_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_2': nn.ReLU(inplace=True),
                'conv_5_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_3': nn.ReLU(inplace=True),
                'maxp_5_3': nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            }))

        self.fc = nn.ModuleDict(OrderedDict(
            {
                'fc6': nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                'fc7': nn.Linear(in_features=4096, out_features=4096),
                'fc8': nn.Linear(in_features=4096, out_features=2622),
            }))

    # def load...

    def forward(self, x):
        pass
