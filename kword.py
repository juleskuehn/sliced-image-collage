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
from kword_utils import chop_charset, resizeTarget

args = sys.argv

# Hardcoding the charset params for convenience
# sourceFn = 'marker-shapes.png'
# slicesX = 12
# slicesY = 2
# xPad = 0
# yPad = 0
sourceFn = 'sc-3tone.png'
slicesX = 50
slicesY = 34
xPad = 0
yPad = 0
# sourceFn = 'hermes-darker.png'
# slicesX = 79
# slicesY = 7
# xPad = 4
# yPad = 4
targetFn = args[1]
rowLength = int(args[2])
c = 1
shrinkX = int(args[3])
shrinkY = int(args[3])
mode = args[4]
gamma = float(args[5])
resume = args[6] if len(args) > 6 and args[6] not in ['save','rs','ri'] else None
# resume = None
numAdjust = 1
randomInit = 'ri' in args
randomOrder = 'ro' in args

print(args)

# dither = 'dither' in args
show = not 'save' in args

#################
# Prepare charset
targetImg = cv2.imread(targetFn, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", targetImg.shape)
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
mockupFn = f"mockup/mp_{targetFn.split('.')[-2][1:]}_{rowLength}_{mode}"

######################
# Prepare target image
shrunkenTarget, shrunkenTargetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).shrunken.shape, (xChange, yChange))
print('shrunken char shape', charSet.get(0).shrunken.shape)
# resizedCharShape = charSet.get(0).shrunken.shape[0] * shrinkY, charSet.get(0).shrunken.shape[1] * shrinkX
resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange))
print('shrunkenTarget.shape', shrunkenTarget.shape)
print('resizedTarget.shape', resizedTarget.shape)


# Save characters
import os
d = os.getcwd() + '\\chars'
filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
for f in filesToRemove:
    os.remove(f) 
for i, char in enumerate(charSet.getSorted()):
    cv2.imwrite('chars/'+str(i)+'.png', char.cropped)


#################################################
# Generate mockup (the part that really matters!)
generator = Generator(resizedTarget, shrunkenTarget, charSet, targetShape=targetImg.shape,
                                    targetPadding=targetPadding, shrunkenTargetPadding=shrunkenTargetPadding)

if resume is not None:
    generator.load_state(resume)

# THIS IS THE LINE THAT MATTERS
generator.generateLayers(compareMode=mode, numAdjustPasses=numAdjust, gamma=gamma, 
                        show=show, mockupFn=mockupFn, randomInit=randomInit,
                        randomOrder=randomOrder)
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
