""" Gonzalo Muradás Odriozola

One click, how convenient.

Script will use DeepGazeIIE to predict saliency maps for all images in INPUT_PATH,
OUTPUT_PATH will be filled with heatmaps (preceaded by 'heatmap_') and their
respective normalized arrays (preceaded by 'mask_').

16/11/2023"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os import listdir
from scipy.ndimage import zoom
from os.path import isfile, join
from scipy.special import logsumexp

import deepgaze_pytorch

# Directories
CENTERBIAS_PATH = 'centerBias/centerbias_mit1003.npy'
INPUT_PATH = 'images/images_art'
OUTPUT_PATH = 'outputArrays'

# GPU (?)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("USING:",DEVICE)

# Downloading model
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

# Reading all files
files = [f for f in listdir(INPUT_PATH) if isfile(join(INPUT_PATH,f))]

# Loading centerbias
centerbias_template = np.load('centerBias/centerbias_mit1003.npy')

# Loop
for image in tqdm(files, total = len(files)):  
    # Loading image
    img = cv2.imread(join(INPUT_PATH, image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Adjusting centerbias to size of image
    centerbias = zoom(centerbias_template, (img.shape[0]/centerbias_template.shape[0], img.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
    centerbias -= logsumexp(centerbias)

    # Predict
    image_tensor = torch.tensor(np.array([img.transpose(2, 0, 1)])).to(DEVICE)
    centerbias_tensor = torch.tensor(np.array([centerbias])).to(DEVICE)
    log_density_prediction = model(image_tensor, centerbias_tensor)
    saliency_map = log_density_prediction.detach().cpu().numpy()[0][0]

    # Transform
    normalized = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blur = cv2.GaussianBlur(normalized,(15,15), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_RAINBOW)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    # Saving image
    cv2.imwrite(join(OUTPUT_PATH, f'heatmap_{image}'),super_imposed_img)
    cv2.imwrite(join(OUTPUT_PATH, f'mask_{image}'),normalized)