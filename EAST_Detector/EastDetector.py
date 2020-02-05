import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
from Dataset import *
from Architecture import PVAnet

def main():
    batch_size = 2
    training_path = "./Samples/"
    transform = transforms.Compose([
                    transforms.Resize(192),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    training_data = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True,  num_workers=4)

    net = PVAnet()

    for i, train in enumerate(training_data, 0):
        inputs, labels = train
        outputs = net.forward(inputs.unsqueeze(0))

if __name__ == '__main__':
    main()
