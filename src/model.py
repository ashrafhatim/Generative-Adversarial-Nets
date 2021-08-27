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


class Generator_1(nn.Module):
  def __init__(self, z_dim, img_shape):
    super(Generator_1, self).__init__()

    self.z_dim = z_dim 
    self.img_shape = img_shape

    self.n_features = self.z_dim
    self.n_out = np.prod(self.img_shape)

    self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.LeakyReLU(0.2)
        )
    self.hidden1 = nn.Sequential(            
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2)
    )
    self.hidden2 = nn.Sequential(
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2)
    )
    
    self.out = nn.Sequential(
        nn.Linear(1024, self.n_out),
        nn.Tanh()
    )

  def forward(self, x):
      x = self.hidden0(x)
      x = self.hidden1(x)
      x = self.hidden2(x)
      x = self.out(x)

      x = x.reshape(-1, *self.img_shape)
      return x
    
class Generator_2(nn.Module):
  def __init__(self, z_dim, img_shape):
    super(Generator_2, self).__init__()
    def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

    self.z_dim = z_dim 
    self.img_shape = img_shape

    self.model = nn.Sequential(
        *block(self.z_dim, 128, normalize=False),
        *block(128, 256),
        *block(256, 512),
        *block(512, 1024),
        nn.Linear(1024, np.prod(self.img_shape)),
        nn.Tanh()
    )

  def forward(self, z):
      img = self.model(z)
      img = img.view(img.size(0), *self.img_shape)
      return img

class Generator_3(nn.Module):
  def __init__(self, z_dim, img_shape):
    super(Generator_3, self).__init__()

    self.z_dim = z_dim                                                          
    self.img_shape = img_shape                                                 
    self.img_dim = np.prod(self.img_shape)                                     
    
    self.layers = nn.Sequential(
        nn.Linear(self.z_dim, 128),
        nn.BatchNorm1d(128, 0.8),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024, self.img_dim),
        nn.Tanh(),
    )

  def forward(self, z):
    z = self.layers(z)
    img = z.reshape(-1, *self.img_shape)

    return img

class Discriminator_1(nn.Module):
  def __init__(self, img_shape):
    super(Discriminator_1, self).__init__()

    self.img_shape = img_shape
    self.in_dim = np.prod(img_shape)  
    self.n_features = self.in_dim
    self.n_out = 1

    self.hidden0 = nn.Sequential( 
            nn.Linear(self.n_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
    self.hidden1 = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        # nn.Dropout(0.3)
    )
    self.hidden2 = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        # nn.Dropout(0.3)
    )
    self.out = nn.Sequential(
        torch.nn.Linear(256, self.n_out),
        # torch.nn.Sigmoid()
    )

  def forward(self, x):
      x = x.reshape(-1, self.in_dim)
      x = self.hidden0(x)
      x = self.hidden1(x)
      x = self.hidden2(x)
      x = self.out(x)
      return x

class Discriminator_2(nn.Module):
  def __init__(self, img_shape):
    super(Discriminator_2, self).__init__()

    self.in_dim = np.prod(img_shape)                                      

    self.layers = nn.Sequential(
        nn.Linear(self.in_dim, 512),
        maxout_mlp(num_units=2, hidden_dim=512),
        nn.Dropout(0.25),
        nn.Linear(512, 256),
        maxout_mlp(num_units=2, hidden_dim=256),
        nn.Dropout(0.25),
        nn.Linear(256, 128),
        maxout_mlp(num_units=2, hidden_dim=128),
        nn.Dropout(0.25),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )

  def forward(self, x):
    x = x.reshape(-1, self.in_dim)
    out = self.layers(x)
    return out

class Discriminator_3(nn.Module):
  def __init__(self, img_shape):
    super(Discriminator_3, self).__init__()

    self.img_shape = img_shape
    self.in_dim = np.prod(img_shape) 

    self.model = nn.Sequential(
        nn.Linear(self.in_dim , 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

  def forward(self, img):
      img_flat = img.view(img.size(0), -1)
      validity = self.model(img_flat)

      return validity

# reference: https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb

class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class maxout_mlp(nn.Module):
    def __init__(self, num_units=2, hidden_dim = 784):
        super(maxout_mlp, self).__init__()
        self.fc1_list = ListModule(self, "fc1_")
        self.fc2_list = ListModule(self, "fc2_")
        self.hidden_dim = hidden_dim
        for _ in range(num_units):
            self.fc1_list.append(nn.Linear(self.hidden_dim, 1024))
            self.fc2_list.append(nn.Linear(1024, self.hidden_dim))

    def forward(self, x): 
        x = x.view(-1, self.hidden_dim)
        x = self.maxout(x, self.fc1_list)
        x = F.dropout(x, training=self.training)
        x = self.maxout(x, self.fc2_list)
        return F.log_softmax(x)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output