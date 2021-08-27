import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter # tensorboard

import argparse

from utils import get_model, train, helper
from model import *

if __name__=="__main__":

    parser = argparse.ArgumentParser()
      
    parser.add_argument("--exp-num", default= 1, type= int)
    parser.add_argument("--epochs", default= 5, type= int)
    parser.add_argument("--TENSORBOARD-PATH", default="/content/gdrive/MyDrive/gans_images/_tensorboard", type= str)
    parser.add_argument("--SAVE-PATH", default="/content/gdrive/MyDrive/gans_images/" , type= str)

    args = parser.parse_args()


# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [ transforms.Resize([28,28]), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=64,
    shuffle=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
exp_num = args.exp_num 
epochs = args.epochs
TENSORBOARD_PATH = args.TENSORBOARD-PATH # dataset folder
SAVE_PATH = args.SAVE-PATH # save folder


generator, discriminator, optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler = get_model.get_model(1, 1, 100, (1,28,28), lr = 5e-5)
train.train(generator, discriminator, optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler , epochs=epochs, dataloader=dataloader, save_path = SAVE_PATH, tensorboard_path = TENSORBOARD_PATH)

helper.sample_img(generator)
