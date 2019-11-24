import torch
import torch.nn as nn
from collections import OrderedDict


class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()

        self.features = nn.ModuleDict(OrderedDict(
            {
                # ============================ B_1 ============================
                'conv_1_1': nn.Conv2d(3, 64, kernel_size=3, padding=1),
                'conv_1_1-relu': nn.ReLU(inplace=True),
                'conv_1_2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
                'conv_1_2-relu': nn.ReLU(inplace=True),
                'conv_1_2-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                # ============================ B_2 ============================
                'conv_2_1': nn.Conv2d(64, 128, kernel_size=3, padding=1),
                'conv_2_1-relu': nn.ReLU(inplace=True),
                'conv_2_2-conv': nn.Conv2d(128, 128, kernel_size=3, padding=1),
                'conv_2_2-relu': nn.ReLU(inplace=True),
                'conv_2_2-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                # ============================ B_3 ============================
                'conv_3_1': nn.Conv2d(128, 256, kernel_size=3, padding=1),
                'conv_3_1-relu': nn.ReLU(inplace=True),
                'conv_3_2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
                'conv_3_2-relu': nn.ReLU(inplace=True),
                'conv_3_3': nn.Conv2d(256, 256, kernel_size=3, padding=1),
                'conv_3_3-relu': nn.ReLU(inplace=True),
                'conv_3_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                # ============================ B_4 ============================
                'conv_4_1': nn.Conv2d(256, 512, kernel_size=3, padding=1),
                'conv_4_1-relu': nn.ReLU(inplace=True),
                'conv_4_2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                'conv_4_2-relu': nn.ReLU(inplace=True),
                'conv_4_3': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                'conv_4_3-relu': nn.ReLU(inplace=True),
                'conv_4_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                # ============================ B_5 ============================
                'conv_5_1': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                'conv_5_1-relu': nn.ReLU(inplace=True),
                'conv_5_2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                'conv_5_2-relu': nn.ReLU(inplace=True),
                'conv_5_3': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                'conv_5_3-relu': nn.ReLU(inplace=True),
                'conv_5_3-maxp': nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            }))

        self.fc = nn.ModuleDict(OrderedDict(
            {
                'fc6': nn.Linear(512 * 7 * 7, 4096),
                'fc7': nn.Linear(4096, 4096),
                'fc8': nn.Linear(4096, 2622),
            }))

    def forward(self, x):
        pass
