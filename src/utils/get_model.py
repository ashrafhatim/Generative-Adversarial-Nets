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
from utils import get_model, train, helper


def get_model(g_num, d_num, z_dim, img_shape, lr = 5e-5):
    def get_generator(num, z_dim, img_shape):
      if num == 1:
        return Generator_1(z_dim, img_shape)
      elif num == 2:
        return Generator_2(z_dim, img_shape)
      elif num == 3:
        return Generator_3(z_dim, img_shape)
      else:
        print("choose a valid generator number next time !")

    def get_discriminator(num, img_shape):
      if num == 1:
        return Discriminator_1(img_shape)
      elif num == 2:
        return Discriminator_2(img_shape)
      elif num == 3:
        return Discriminator_3(img_shape)
      else:
        print("choose a valid discriminator number next time !")

    def get_gan(g_num, d_num, z_dim, img_shape ):
      generator = get_generator(g_num, z_dim, img_shape)
      discriminator = get_discriminator(d_num, img_shape)  
      generator.to(device)
      discriminator.to(device)
      return generator, discriminator

    def get_optimizers(generator, discriminator, lr = 5e-5):
      optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
      optimizer_G_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
      #torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) #torch.optim.SGD(generator.parameters(), lr=0.0002, momentum=0.9) #torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
      
      optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
      optimizer_D_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
      #torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)) #torch.optim.SGD(discriminator.parameters(), lr=0.0002, momentum=0.9) #torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
      return optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler


    generator, discriminator =  get_gan(g_num, d_num, z_dim, img_shape )
    optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler = get_optimizers(generator, discriminator, lr = lr)

    return generator, discriminator, optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler