import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import skimage as ski
import skimage.exposure as ski_exposure


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.chosen_features = [0, 5, 10, 19, 28]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if layer_num in self.chosen_features:
                features.append(x)

        return features
