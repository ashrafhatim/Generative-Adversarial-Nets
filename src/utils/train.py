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


def train(generator, discriminator, optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler, dataloader, z_dim=100, epochs=200, save_path = SAVE_PATH, tensorboard_path = TENSORBOARD_PATH):
  # launch tensorboard
  sw = SummaryWriter(tensorboard_path)

  # training loop
  for epoch in range(epochs):
      for i, (imgs, _) in enumerate(dataloader):

          # Adversarial ground truths
          valid = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(0.9), requires_grad=False).to(device)
          fake = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device)

          # Configure input
          real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)

          ## Discriminator training
          for _ in range(5):
            optimizer_D.zero_grad()

            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], z_dim)))).to(device)
            gen_imgs = generator(z).detach()

            D_real = discriminator(real_imgs)
            D_fake = discriminator(gen_imgs)

            d_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            d_loss.backward()
            optimizer_D.step() 

            for p in discriminator.parameters():
              p.data.clamp_(-0.01, 0.01)

          ## Generator training
          optimizer_G.zero_grad()

          z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], z_dim)))).to(device)
          gen_imgs_g = generator(z)

          D_fake = discriminator(gen_imgs_g)

          g_loss = -torch.mean(D_fake)

          g_loss.backward()
          optimizer_G.step() 

          print(
              "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]".format(epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
          )

          batches_done = epoch * len(dataloader) + i
          if batches_done % 100 == 0:
              save_image(gen_imgs.data[:25], save_path + "%d.png" % batches_done, nrow=5, normalize=True)

          sw.add_scalars("Loss", {"g_loss":g_loss, "d_loss":d_loss}, batches_done)

      optimizer_G_scheduler.step()
      optimizer_D_scheduler.step()

      if epoch % 1 == 0:
        torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_G_scheduler_state_dict': optimizer_G_scheduler.state_dict(),
                'optimizer_D_scheduler_state_dict': optimizer_D_scheduler.state_dict(),
                }, save_path+"models/" +str(epoch) +"epochs" + ".pt")


  torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'optimizer_G_scheduler_state_dict': optimizer_G_scheduler.state_dict(),
            'optimizer_D_scheduler_state_dict': optimizer_D_scheduler.state_dict(),
            }, save_path+ "models/" +str(epochs) +"epochs" + ".pt")
