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

from selector_helpers import *

import sys
args = sys.argv

print("""Usage: python dither_selector.py sourceImg targetImg slicesX slicesY rowLength bestK ditherMode [noblank]
    sourceImg, targetImg: file paths
    slicesX, slicesY: positive integers
    rowLength: best set to same as slicesX, for testing reconstruction of the same sourceImg vs targetImg
    bestK: how many nearest neighbours are given to the MSE evaluator
    ditherMode: one of ["fs", "li", "priority", "none"]
    noblank: this word will override the default behaviour, which is to keep a blank space in the charset.
    """)

if len(args) < 7:
    exit()

sourceImg = args[1]
targetImg = args[2]
slicesX = int(args[3])
slicesY = int(args[4])
rowLength = int(args[5])
kBest = float(args[6])
ditherMode = 'none'

blankSpace = not 'noblank' in args

target = cv2.imread(targetImg, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", target.shape)

# Get character set and related info
cropped, padded, (xPad, yPad), (xChange, yChange) = chop_charset(
    fn=sourceImg, numX=slicesX, numY=slicesY, startX=0, startY=0,
    xPad=0, yPad=0, shrink=4, blankSpace=blankSpace)

def composite6Mult(charTL, charTC, charTR, charBL, charBC, charBR):
    charHeight, charWidth = charTL.shape
    halfHeight, halfWidth = charHeight//2, charWidth//2

    imgData = np.full((charHeight+halfHeight, charWidth*2), 1.0, dtype="float32")
    imgData[:charHeight, :charWidth] = charTL
    imgData[:charHeight, halfWidth:charWidth+halfWidth] *= charTC
    imgData[:charHeight, charWidth:] *= charTR
    imgData[halfHeight:, :charWidth] *= charBL
    imgData[halfHeight:, halfWidth:charWidth+halfWidth] *= charBC
    imgData[halfHeight:, charWidth:] *= charBR
        
    return np.array(imgData * 255, dtype='uint8')


# Generate combos
def gen_combos6Mult(charset):
    inv = []
    # Invert colors
    for char in charset:
        inv.append(np.array(char / 255, dtype="float32"))

    charHeight, charWidth = charset[0].shape
    halfHeight, halfWidth = charHeight//2, charWidth//2

    combos = [] # [ index, data ]
    for i, charTL in enumerate(inv):
        for j, charTC in enumerate(inv):
            for k, charTR in enumerate(inv):
                for l, charBL in enumerate(inv):
                    for m, charBC in enumerate(inv):
                        for n, charBR in enumerate(inv):
                            # Trivial compositing
                            imgData = composite6Mult(charTL, charTC, charTR, charBL, charBC, charBR)
                            combos.append(imgData[halfHeight:charHeight, halfWidth:charWidth+halfWidth])

    print(len(combos))
    return combos

# 8: 0
# 9: -
# 10: =
# 19: o
# 21: 1/2
# 30: k
# 33: `
# 35: x
# 41: ,
# 44: #
# 45: $
# 47: _
# 48: &
# 49: '
# 50: (
# 51: )
# 65: @
# 71: H
# 75: :
# 76: ^
# 78: X
# 84: ?
# 85: c with thing
# 86: .
# bestChars = cropped[[8, 19, 33, 35, 86, 44, 45, 65, 71, -1]]
bestChars = cropped[[65, 19, 44, -1]]

# Takes around 5GB of ram and 20 seconds for the 1,000,000 images with addition
# Takes around 60 seconds with multiplication (much better results)
combos = gen_combos6Mult(bestChars)

cv2.imwrite('tfile.png', combos[0])

# Dimensions of a single char
charHeight, charWidth = combos[0].shape
dim = charWidth * charHeight

# levelAdjustedChars = cropped

# angularNN = AnnoyIndex(dim, metric='angular')
# angularNN.load('angular.ann')
# euclideanNN = AnnoyIndex(dim, metric='euclidean')
# euclideanNN.load('euclidean.ann')

angularNN = buildModel(dim, combos, 'angular')
euclideanNN = buildModel(dim, combos, 'euclidean')

# Resize target photo to rowLength * charWidth and pad to next multiple of charHeight
rphoto, targetPadding = resizePhoto(target, rowLength, (charWidth, charHeight), (xChange, yChange))

# This is where the magic happens! Choose slices from source to represent target
t = genTypable(rphoto, combos[0].shape, angularNN, euclideanNN, kBest, combos)

# Generate  mockup (reconstruction of target in terms of source)
m = genMockup(t, combos, (target.shape[1], target.shape[0]), targetPadding)

mockupFn = f"mockup/mockup_{sourceImg.split('.')[-2][1:]}_{targetImg.split('.')[-2][1:]}_{rowLength}w__best{kBest}.png"
print("writing file:")
print(mockupFn)
cv2.imwrite(mockupFn, m)