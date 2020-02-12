import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
from Architecture import *

import logging

def main():
    logging.basicConfig(filename="CreluTest.log", level=logging.INFO)
    logging.info("Started")
    batch_size = 2
    training_path = "./Samples/"
    transform = transforms.Compose([
                    transforms.Resize(192),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    training_data = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True,  num_workers=4)

    net = CRelu()

    for i, train in enumerate(training_data, 0):
        logging.info("Image #{}".format(i))
        inputs, labels = train
        outputs = net.forward(inputs.unsqueeze(0))
        #logging.info("Output of CRelu:\n{}".format(outputs))
        logging.info("\n")

    logging.info("Finished")

if __name__ == '__main__':
    main()
