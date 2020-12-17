import os
import sys
from pathlib import Path
from typing import List

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

DIR_PATH = Path(__file__).parent
sys.path.append(str(DIR_PATH))
import utils
import vgg
import painting_extractor as pe


# Hyperparameters
total_steps = 6000
log_freq = 200
learning_rate = 1e-3
alpha = 1  # content loss
beta = 1e-2  # how much style

# Style Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((356, 356))
    , transforms.ToTensor()
    # , transforms.Normalize(0.5, 0.5)
])
model = vgg.VGG().to(device).eval()


def apply_style_on_painting(painting, style):
    if isinstance(painting, np.ndarray):
        painting = ski_exposure.rescale_intensity(painting, out_range=(0., 1.))
        painting = ski.img_as_ubyte(painting)
        painting = Image.fromarray(painting)

    if isinstance(style, np.ndarray):
        style = ski_exposure.rescale_intensity(style, out_range=(0., 1.))
        style = ski.img_as_ubyte(style)
        style = Image.fromarray(style)

    print(painting)
    print(style)

    painting, painting_orig_shape = utils.load_img(painting, transform, device, True)
    style = utils.load_img(style, transform, device)
    gen_painting = painting.clone().requires_grad_(True)
    
    optimizer = optim.Adam([gen_painting], lr=learning_rate)

    for step in range(total_steps):
        orig_features = model(painting)
        style_features = model(style)
        gen_features = model(gen_painting)

        orig_loss = style_loss = 0
        for orig_feature, style_feature, gen_feature in zip(
            orig_features
            , style_features
            , gen_features
        ):

            batch_size, c, h, w = gen_feature.shape
            orig_loss += torch.mean((gen_feature - orig_feature) ** 2)

            gen_gram = gen_feature.view(c, h * w).mm(gen_feature.view(c, h * w).t())
            style_gram = style_feature.view(c, h * w).mm(style_feature.view(c, h * w).t())
            style_loss += torch.mean((gen_gram - style_gram) ** 2)

        total_loss = alpha * orig_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % log_freq == 0:
            gen_painting_numpy = gen_painting.cpu().detach().numpy()[0]
            gen_painting_numpy = np.moveaxis(gen_painting_numpy, 0, -1)
            gen_painting_numpy = cv.resize(gen_painting_numpy, painting_orig_shape)
            gen_painting_numpy = ski_exposure.rescale_intensity(gen_painting_numpy, out_range=(0., 1.))

            yield step, total_loss.item(), gen_painting_numpy
