import os
import sys
from pathlib import Path
from typing import List
import argparse

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
import painting_extractor as pe
import style_transfer as st
import utils

# parser = argparse.ArgumentParser(description='Apply styles to image.')
# parser.add_argument('--orig', type=str, help='Original images dirpath.', default='dataset/orig')
# parser.add_argument('--style', type=str, help='Style images dirpath.', default='dataset/style')
# parser.add_argument('--gen', type=str, help='Generated images dirpath.', default='dataset/gen')
# args = parser.parse_args()

shots_dirpath = 'dataset/orig'  # args.orig
styles_dirpath = 'dataset/style'  # args.style
gen_dirpath = 'dataset/gen'  # args.gen


def extract_painting_apply_style(img_path: str, style_path: str):
    orig_scene = cv.imread(img_path)
    orig_scene = cv.cvtColor(orig_scene, cv.COLOR_BGR2RGB)
    style = cv.imread(style_path)
    style = cv.cvtColor(style, cv.COLOR_BGR2RGB)

    # orig_painting, orig_painting_pts = pe.extract_painting(orig_scene.copy())
    orig_painting = orig_scene.copy()

    for step, loss, gen_painting in st.apply_style_on_painting(orig_painting, style):
        # gen_scene = pe.inject_back(gen_painting.copy(), orig_painting_pts, orig_scene)
        gen_scene = gen_painting.copy()
        yield step, loss, gen_painting, gen_scene


def main(shots_dirpath, styles_dirpath, gen_dirpath):
    shots_names = os.listdir(shots_dirpath)
    styles_names = os.listdir(styles_dirpath)

    for shot_name in shots_names:
        # shot_name = 'trainw1.jpg'
        shot_path = os.path.join(shots_dirpath, shot_name)

        for style_name in styles_names:
            # style_name = 'blue.jpg'
            style_path = os.path.join(styles_dirpath, style_name)

            print('Generating painting...')
            print(f'Original: {shot_name}')
            print(f'Style: {style_name}')
        
            for step, loss, gen_painting, gen_scene in extract_painting_apply_style(shot_path, style_path):
                print(step, loss)
                plt.imsave(
                    os.path.join(gen_dirpath, f'{Path(shot_name).stem}_{Path(style_name).stem}.png')
                    , gen_painting
                )
                plt.imsave(
                    os.path.join(gen_dirpath, f'{Path(shot_name).stem}_{Path(style_name).stem}_scene.png')
                    , gen_scene
                )


if __name__ == "__main__":
    main(shots_dirpath, styles_dirpath, gen_dirpath)
