from typing import Callable, Union

import torch
import torchvision.transforms as transforms
from PIL import Image


def load_img(
    img
    , transform: Callable=transforms.Compose([
        transforms.Resize((356, 356))
        , transforms.ToTensor()
        # , transforms.Normalize(0.5, 0.5)
    ])
    , device: torch.device=torch.device('cpu')
    , return_shape: bool=False
):

    if isinstance(img, str):
        img = Image.open(img)

    img_shape = None
    if return_shape:
        img_shape = img.size[-2:]

    img = transform(img).unsqueeze(0)
    img = img.to(device)

    if img_shape is None:
        output = img
    else:
        output = img, img_shape

    return output
