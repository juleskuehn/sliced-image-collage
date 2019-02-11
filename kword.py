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

# Hardcoding the charset params for convenience
sourceFn = 'hermes-darker.png'
targetFn = args[1]
slicesX = 79
slicesY = 7
rowLength = int(args[4])
c = 1
shrinkX = 1
shrinkY = 1
modes = args[2]
gamma = float(args[3])
numAdjust = 1

print(args)

dither = 'dither' in args
show = not 'save' in args

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
mockupFn = f"mockup/mp_{targetFn.split('.')[-2][1:]}_{rowLength}_{modes}"

######################
# Prepare target image
resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, cropped[0].shape, (xChange, yChange))


#################################################
# Generate mockup (the part that really matters!)
generator = Generator(resizedTarget, charSet, targetShape=targetImg.shape,
                                    targetPadding=targetPadding)

# THIS IS THE LINE THAT MATTERS
generator.generateLayers(compareModes=modes, numAdjustPasses=numAdjust,
                        show=show, mockupFn=mockupFn, gamma=gamma,
                        randomOrder=False, randomInit=True)
# THIS IS THE LINE THAT MATTERS


###################
# Save mockup image
print("writing file:",mockupFn)
mockupImg = generator.mockupImg
if targetPadding > 0: # Crop and resize mockup to match target image
    mockupImg = mockupImg[:-targetPadding, :]
resized = cv2.resize(mockupImg, dsize=(targetImg.shape[1],targetImg.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imwrite(mockupFn+'.png', resized)

############################
# Calculate scores on result
print("PSNR:", compare_psnr(resized, targetImg))
print("SSIM:", compare_ssim(resized, targetImg))

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