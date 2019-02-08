import numpy as np
import matplotlib.pyplot as plt
import math
import random
from timeit import Timer
from annoy import AnnoyIndex
from matplotlib.image import imread, imsave
from PIL import Image
import cv2
from math import inf
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr
import sys
from scipy.ndimage import sobel

from combo import Combo, ComboSet
from combo_grid import ComboGrid
from char import Char, CharSet
from generator import Generator
from kword_util import *

args = sys.argv

sourceFn = args[1]
targetFn = args[2]
slicesX = int(args[3])
slicesY = int(args[4])
rowLength = int(args[5])
c = int(args[6])
shrinkX = int(args[7])
shrinkY = int(args[8])
numAdjust = int(args[9])
dither = 'dither' in args
preview = 'preview' in args

#################
# Prepare charset
targetImg = cv2.imread(targetFn, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", targetImg.shape)
xPad = 4
yPad = 4
cropped, padded, (xCropPos, yCropPos), (xChange, yChange) = chop_charset(
    fn=sourceFn, numX=slicesX, numY=slicesY, startX=0, startY=0,
    xPad=xPad, yPad=yPad, shrinkX=shrinkX, shrinkY=shrinkY, blankSpace=True)
cropSettings = {
    'xPad': xPad,
    'yPad': yPad,
    'xCropPos': xCropPos,
    'yCropPos': yCropPos,
    'shrinkX': shrinkX,
    'shrinkY': shrinkY
}
charSet = CharSet(padded, cropSettings)

######################
# Prepare target image
resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, cropped[0].shape, (xChange, yChange))


#################################################
# Generate mockup (the part that really matters!)
generator = Generator(resizedTarget, charSet, targetShape=targetImg.shape,
                                    targetPadding=targetPadding)
generator.generateLayers(compareModes=['blend','mse','mse','mse'], numAdjustPasses=numAdjust)


###################
# Save mockup image
mockupFn = f"mockup/mp_{sourceFn.split('.')[-2][1:]}_{targetFn.split('.')[-2][1:]}_{rowLength}"
print("writing file:",mockupFn)
mockupImg = generator.mockupImg
if targetPadding > 0: # Crop and resize mockup to match target image
    mockupImg = mockupImg[:-targetPadding, :]
resized = cv2.resize(mockupImg, dsize=(targetImg.shape[1],targetImg.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imwrite(mockupFn+'.png', resized)

# Overlay the original image for comparison
# cv2.imwrite(mockupFn+'c.png', cv2.addWeighted(resized,0.5,targetImg,0.5,0))

# # Save characters
# import os
# d = os.getcwd() + '\\chars'
# filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
# for f in filesToRemove:
#     os.remove(f) 
# for i, char in enumerate(padded):
#     cv2.imwrite('chars/padded_'+str(i)+'.png', char)