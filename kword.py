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
from kword_utils import chop_charset, resizeTarget, genMockup

args = sys.argv

# Hardcoding the charset params for convenience

# sourceFn = 'marker-shapes.png'
# slicesX = 12
# slicesY = 2
# xPad = 0
# yPad = 0

sourceFn = 'sc-3toneNew.png'
slicesX = 45
slicesY = 21
xPad = 0
yPad = 0

# sourceFn = 'sc-3toneNew2.png'
# slicesX = 45
# slicesY = 25
# xPad = 0
# yPad = 0

# sourceFn = 'sc-3tone.png'
# slicesX = 50
# slicesY = 34
# xPad = 0
# yPad = 0

# sourceFn = 'sc-1tone.png'
# slicesX = 26
# slicesY = 15
# xPad = 0
# yPad = 0

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
resume = args[6] if len(args) > 6 and args[6] not in ['save','ro','ri','autocrop','crop'] else None
# resume = None
numAdjust = 1
randomInit = 'ri' in args
randomOrder = 'ro' in args
autoCrop = 'autocrop' in args
crop = 'crop' in args
zoom = 0
shiftLeft = 0
shiftUp = 0
if args[6] == 'crop':
    zoom = int(args[7])
    shiftLeft = int(args[8])
    shiftUp = int(args[9])
if len(args) > 10:
    resume = args[10]

print(args)

# dither = 'dither' in args
show = not 'save' in args

#################
# Prepare charset
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


targetImg = cv2.imread(targetFn, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", targetImg.shape)

# Save characters
# import os
# d = os.getcwd() + '\\chars'
# filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
# for f in filesToRemove:
#     os.remove(f) 
# for i, char in enumerate(charSet.getSorted()):
#     cv2.imwrite('chars/'+str(i)+'.png', char.cropped)



# Autocrop routine, terminates the rest of the program
if autoCrop:
    annSaved = False
    charHeight, charWidth = charSet.get(0).cropped.shape

    scores = []
    # Try 2 zooms: none, and add 1 quadrant to bottom and right sides
    for zoom in range(4):
        # Shift 0-1.5 quadrants
        for shiftLeft in range(4):
            shiftLeft = int(((charWidth / 4) * shiftLeft) / 2)
            for shiftUp in range(4):
                shiftUp = int(((charHeight / 4) * shiftUp) / 2)
                resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange))
                origShape = resizedTarget.shape
                height = (resizedTarget.shape[0] + zoom * charHeight//4)
                width = (resizedTarget.shape[1] + zoom * charWidth//4)
                # zoom target
                resizedTarget = cv2.resize(resizedTarget, dsize=(width, height), interpolation=cv2.INTER_AREA)
                # shift target left and up
                resizedTarget = resizedTarget[shiftUp:,shiftLeft:]
                newTarget = np.full(origShape, 255, dtype='uint8')
                # crop or pad
                if resizedTarget.shape[0] >= origShape[0]:
                    if resizedTarget.shape[1] >= origShape[1]:
                        # crop right and bottom
                        newTarget[:, :] = resizedTarget[:origShape[0], :origShape[1]]
                        minShape = newTarget.shape
                    # pad right, crop bottom
                    else:
                        newTarget[:, :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
                        minShape = [origShape[0], resizedTarget.shape[1]]
                else:
                    if resizedTarget.shape[1] >= origShape[1]:
                        # crop right, pad bottom
                        newTarget[:resizedTarget.shape[0], :] = resizedTarget[:, :origShape[1]]
                        minShape = [resizedTarget.shape[0], origShape[1]]
                    else:
                        # pad right and bottom
                        newTarget[:resizedTarget.shape[0], :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
                        minShape = resizedTarget.shape
                #################################################
                # Generate mockup (the part that really matters!)
                generator = Generator(newTarget, newTarget, charSet, targetShape=targetImg.shape,
                                                    targetPadding=targetPadding, shrunkenTargetPadding=targetPadding)
                if annSaved:
                    generator.loadAnn()
                else:
                    # Build angular and euclidean ANN models
                    generator.buildAnn()
                    annSaved = True
                # THIS IS THE LINE THAT MATTERS
                generator.generateLayers(compareMode=mode, numAdjustPasses=numAdjust, gamma=gamma, 
                                    show=show, mockupFn=mockupFn, init='blend', initOnly=True)
                ###################
                # Save init image
                mockupFn = f'mockup/init_zoom{zoom}_left{shiftLeft}_up{shiftUp}'
                print("writing init file: ",mockupFn)
                mockupImg = generator.mockupImg
                # Crop added whitespace from shifting
                mockupImg = mockupImg[:minShape[0], :minShape[1]]
                newTarget = newTarget[:minShape[0], :minShape[1]]
                psnr = compare_psnr(mockupImg, newTarget)
                ssim = compare_ssim(mockupImg, newTarget)
                print("PSNR:", psnr)
                print("SSIM:", ssim)
                cv2.imwrite(mockupFn+'.png', mockupImg)
                scores.append((ssim+psnr, ssim, psnr, mockupFn))

    scores = sorted(scores, reverse=True)
    for score in scores:
        print(score)
        
    exit()

# else:
#     shrunkenTarget, shrunkenTargetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).shrunken.shape, (xChange, yChange))
#     print('shrunken char shape', charSet.get(0).shrunken.shape)
#     # resizedCharShape = charSet.get(0).shrunken.shape[0] * shrinkY, charSet.get(0).shrunken.shape[1] * shrinkX
#     resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange))
#     print('shrunkenTarget.shape', shrunkenTarget.shape)
#     print('resizedTarget.shape', resizedTarget.shape)
charHeight, charWidth = charSet.get(0).cropped.shape
resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange))
origShape = resizedTarget.shape
height = (resizedTarget.shape[0] + zoom * charHeight//4)
width = (resizedTarget.shape[1] + zoom * charWidth//4)
# zoom target
resizedTarget = cv2.resize(resizedTarget, dsize=(width, height), interpolation=cv2.INTER_AREA)
# shift target left and up
resizedTarget = resizedTarget[shiftUp:,shiftLeft:]
newTarget = np.full(origShape, 255, dtype='uint8')
# crop or pad
if resizedTarget.shape[0] >= origShape[0]:
    if resizedTarget.shape[1] >= origShape[1]:
        # crop right and bottom
        newTarget[:, :] = resizedTarget[:origShape[0], :origShape[1]]
        minShape = newTarget.shape
    # pad right, crop bottom
    else:
        newTarget[:, :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
        minShape = [origShape[0], resizedTarget.shape[1]]
else:
    if resizedTarget.shape[1] >= origShape[1]:
        # crop right, pad bottom
        newTarget[:resizedTarget.shape[0], :] = resizedTarget[:, :origShape[1]]
        minShape = [resizedTarget.shape[0], origShape[1]]
    else:
        # pad right and bottom
        newTarget[:resizedTarget.shape[0], :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
        minShape = resizedTarget.shape
#################################################
# Generate mockup (the part that really matters!)
generator = Generator(newTarget, newTarget, charSet, targetShape=targetImg.shape,
                                    targetPadding=targetPadding, shrunkenTargetPadding=targetPadding)
# if annSaved:
#     generator.loadAnn()
# else:
#     # Build angular and euclidean ANN models
#     generator.buildAnn()
#     annSaved = True
# #################################################
# # Generate mockup (the part that really matters!)
# generator = Generator(resizedTarget, shrunkenTarget, charSet, targetShape=targetImg.shape,
#                                     targetPadding=targetPadding, shrunkenTargetPadding=shrunkenTargetPadding)

if resume is not None:
    generator.load_state(resume)

# Build angular and euclidean ANN models
generator.buildAnn()
# THIS IS THE LINE THAT MATTERS
generator.generateLayers(compareMode=mode, numAdjustPasses=numAdjust, gamma=gamma, 
                    show=show, mockupFn=mockupFn, init='blend' if resume is None else None)

# print(generator.comboGrid)

###################
# Save mockup image
print("writing file:",mockupFn)
mockupImg = generator.mockupImg
if targetPadding > 0: # Crop and resize mockup to match target image
    mockupImg = mockupImg[:-targetPadding, :]

resized = cv2.resize(mockupImg, dsize=(targetImg.shape[1],targetImg.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imwrite(mockupFn+'.png', resized)

#############
# Save layers
print("saving layers")
# layerNames = ['BR', 'BL', 'TR', 'TL']
# for i, layer in enumerate(generator.comboGrid.getLayers()):
#     layerImg = genMockup(layer, generator, targetImg.shape, targetPadding, crop=False, addFixed=False)
#     cv2.imwrite(mockupFn+'layer'+layerNames[i]+'.png', layerImg)

############################
# Calculate scores on result
print("PSNR:", compare_psnr(resized, targetImg))
print("SSIM:", compare_ssim(resized, targetImg))

# Overlay the original image for comparison
# cv2.imwrite(mockupFn+'c.png', cv2.addWeighted(resized,0.5,targetImg,0.5,0))
