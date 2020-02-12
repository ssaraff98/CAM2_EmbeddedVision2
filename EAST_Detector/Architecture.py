import torch
from torch import nn
import numpy as np

import logging

class CRelu(nn.Module):
    def __init__(self):
        super(CRelu, self).__init__()
        self.activation = nn.ReLU(inplace=True)

    def CRelu_Base(self, x, in_channels, out_channels):
        logging.info("\nCRelu_Base")
        # 1/2 * input size
        convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        #nn.init.xavier_uniform_(convolution1.weight)
        normalization = nn.BatchNorm2d(num_features=out_channels * 2)
        input = convolution1(x)
        logging.info("Input x dimensions after 3x3: [1, channels, height, width] = {}".format(input.size()))
        input = torch.cat((input, -input), 1)
        logging.info("Input x dimensions after concatenation: [1, channels, height, width] = {}".format(input.size()))
        input = normalization(input)
        logging.info("Input x dimensions after normalization: [1, channels, height, width] = {}".format(input.size()))

        convolution2 = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=1)
        input = self.activation(convolution2(input))
        logging.info("Input x dimensions after 1x1: [1, channels, height, width] = {}".format(input.size()))

        # 1/4 * input size
        pool = nn.MaxPool2d(kernel_size=3, stride=2)
        input = pool(input)
        logging.info("Input x dimensions after maxpool: [1, channels, height, width] = {}".format(input.size()))
        return input

    def CRelu_Residual(self, x, in_channels, mid_channels, out_channels, stride=1, projection=False):
        logging.info("\nCRelu_Residual")
        copy_input = x

        convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=stride)
        normalization1 = nn.BatchNorm2d(num_features=mid_channels)
        input = self.activation(normalization1(convolution1(x)))
        logging.info("Input x dimensions after 1x1: [1, channels, height, width] = {}".format(input.size()))

        convolution2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        normalization2 = nn.BatchNorm2d(num_features=mid_channels * 2)
        input = convolution2(input)
        logging.info("Input x dimensions after 3x3: [1, channels, height, width] = {}".format(input.size()))
        input = torch.cat((input, -input), 1)
        logging.info("Input x dimensions after concatenation: [1, channels, height, width] = {}".format(input.size()))
        input = self.activation(normalization2(input))
        logging.info("Input x dimensions after normalization: [1, channels, height, width] = {}".format(input.size()))

        convolution3 = nn.Conv2d(in_channels=mid_channels * 2, out_channels=out_channels, kernel_size=1)
        input = self.activation(convolution3(input))
        logging.info("Input x dimensions after 1x1: [1, channels, height, width] = {}".format(input.size()))

        if projection:
            projectionConvolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
            copy_input = projectionConvolution(copy_input)
            logging.info("Input x dimensions after projection: [1, channels, height, width] = {}".format(copy_input.size()))

        return self.activation(input + copy_input)

    def forward(self, x):
        logging.info("\nInput x dimensions: [1, channels, height, width] = {}\n".format(x.size()))

        # 1/4 * input size
        x1 = self.CRelu_Base(x, in_channels=x.size()[1], out_channels=16)
        logging.info("\nx1 dimensions after base: [1, channels, height, width] = {}\n".format(x1.size()))

        # 1/4 * input size
        x2 = self.CRelu_Residual(x1, in_channels= 32, mid_channels=24, out_channels= 64, stride=1, projection=True)
        x2 = self.CRelu_Residual(x2, in_channels= 64, mid_channels=24, out_channels= 64)
        x2 = self.CRelu_Residual(x2, in_channels= 64, mid_channels=24, out_channels= 64)
        logging.info("\nx2 dimensions after residual1: [1, channels, height, width] = {}\n".format(x2.size()))

        # 1/8 * input size
        x3 = self.CRelu_Residual(x2, in_channels= 64, mid_channels=48, out_channels=128, stride=2, projection=True)
        x3 = self.CRelu_Residual(x3, in_channels=128, mid_channels=48, out_channels=128)
        x3 = self.CRelu_Residual(x3, in_channels=128, mid_channels=48, out_channels=128)
        x3 = self.CRelu_Residual(x3, in_channels=128, mid_channels=48, out_channels=128)
        logging.info("\nx3 dimensions after residual2: [1, channels, height, width] = {}\n".format(x3.size()))

        return x3
