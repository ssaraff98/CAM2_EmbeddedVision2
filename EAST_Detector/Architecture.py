import torch
from torch import nn
import numpy as np

class PVAnet(nn.Module):
    def __init__(self):
        super(PVAnet, self).__init__()

    def CRelu(self, x, c):
        # Convolution 1 - 7x7 kernel, 16 output channels, 1/2 * input size
        convolution1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.xavier_uniform_(convolution1.weight)

        normalization = nn.BatchNorm2d(num_features=16)
        input = normalization(convolution1(x))

        # Concatenation of positive and negative features
        concatenation = torch.cat((input, -input), 1)

        # Convolution 2 - 1x1 kernel, 16 output channels, 1 * input size
        convolution2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, bias=True)
        input = convolution2(input)

        activation = nn.ReLU()
        return activation(input)

    def forward(self, x):
        c = x.size()[1]
        x = self.CRelu(x, c)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        layer1 = pool1(x)
        return x
