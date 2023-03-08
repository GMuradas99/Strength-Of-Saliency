# Python v=3.7.9

import cv2
import torch
import numpy as np

from tqdm import tqdm
from scipy.ndimage import zoom
from os.path import join, exists
from scipy.special import logsumexp

import deepgaze_pytorch

### HELPER FUNCTIONS ###

# Returns the correct path for the image
def getImgPath(imgName):
    if exists(join('images/images_art',imgName)):
        return join('images/images_art',imgName)
    elif exists(join('images/images_nat', imgName)):
        return join('images/images_nat',imgName)
    else:
        return "error" 

### MAIN FUNCTIONS ###

# Saves all predicted saliency maps w
def getSaliencyHeatmaps(images, outputPath):
    # Downloading model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("USING:",DEVICE)
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

    # Loading images
    centerbias_template = np.load('centerBias/centerbias_mit1003.npy')
    for image in tqdm(images, total = len(images)):
        # Loading image
        img = cv2.imread(getImgPath(image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Centerbias
        centerbias = zoom(centerbias_template, (img.shape[0]/centerbias_template.shape[0], img.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
        centerbias -= logsumexp(centerbias)
        # Predict
        image_tensor = torch.tensor(np.array([img.transpose(2, 0, 1)])).to(DEVICE)
        centerbias_tensor = torch.tensor(np.array([centerbias])).to(DEVICE)
        log_density_prediction = model(image_tensor, centerbias_tensor)
        saliency_map = log_density_prediction.detach().numpy()[0][0]
        # Transform
        normalized = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blur = cv2.GaussianBlur(normalized,(15,15), 11)
        heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_RAINBOW)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

        cv2.imwrite(join(outputPath, f'heatmap_{image}'),super_imposed_img)
        cv2.imwrite(join(outputPath, f'mask_{image}'),normalized)