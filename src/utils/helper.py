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


def sample_img(generator):
  z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, 100)))).to(device
                                                                      )
  generator.eval()
  img = generator(z)
  plt.imshow( img[0].cpu().detach().numpy().reshape(28,28), cmap='gray')

def load_generator( Path = "/content/gdrive/MyDrive/gans_images/models/200epochs.pt"):

  generator, discriminator, optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler = get_model.get_model(1, 1, 100, (1,28,28), lr = 5e-5)

  checkpoint = torch.load(Path)
  generator.load_state_dict(checkpoint['generator_state_dict'])

  return generator
