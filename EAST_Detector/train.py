import torch
import torchvision
from torch import nn, optim
import icdar_tools import icdar

def train():
    dataGenerator = icdar.get_batch(num_workers=num_readers,
                                    training_data_path='path/to_data/icdar15/train/',
                                    input_size=input_size,
                                    batch_size=batch_size_per_gpu * len(gpus))
