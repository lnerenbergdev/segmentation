from pathlib import Path
import albumentations as albu
import os
import cv2
import numpy as np
import torch
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from torch import nn
from iglovikov_helper_functions.dl.pytorch.utils import rename_layers
from segmentation_models_pytorch import Unet
from matplotlib import pyplot as plt
from PIL import Image


def test_augmentations(MAX_SIZE=512):
    transform = albu.Compose(
        [albu.LongestMaxSize(max_size=MAX_SIZE), albu.Normalize(p=1)], p=1
    )
    return albu.Compose(transform)

def load_model(model_path):
    
    model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model

def predict(image_path, MAX_SIZE=512):
    path_base, path_suffix = os.path.splitext(image_path)[0:2]
    out_path = path_base + "_processed" + path_suffix

    device = 'cpu'

    model = load_model('unet-timm-efficient-b3.pth')
    model.eval()

    transform = test_augmentations()
    model = model.to(device)

    original_image = np.array(Image.open(image_path))
    original_height, original_width = original_image.shape[:2]

    image = transform(image=original_image)["image"]
    padded_image, pads = pad(image, factor=MAX_SIZE, border=cv2.BORDER_CONSTANT)

    with torch.no_grad():
        x = torch.unsqueeze(tensor_from_rgb_image(padded_image), 0)
        prediction = model(x.to(device))[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    mask = cv2.resize(
        mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )

    # Background image
    bg = np.zeros([original_height, original_width, 3], dtype=np.uint8)
    bg.fill(245) # Can add color select here

    # Filler until I figure out brgb conversions
    og = cv2.imread(image_path)

    # Mask original image with selected background
    mask = cv2.bitwise_and(og, bg, mask=mask)

    # Replace with bucket upload and response with location
    cv2.imwrite(out_path, mask)

    return out_path



