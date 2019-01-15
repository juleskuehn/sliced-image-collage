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

if len(args) < 8:
    exit()

sourceImg = args[1]
targetImg = args[2]
slicesX = int(args[3])
slicesY = int(args[4])
rowLength = int(args[5])
kBest = int(args[6])
ditherMode = args[7]
dist = 'angular'

blankSpace = not 'noblank' in args

target = cv2.imread(targetImg, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", target.shape)

# Get character set and related info
cropped, padded, (xPad, yPad), (xChange, yChange) = chop_charset(
    fn=sourceImg, numX=slicesX, numY=slicesY, startX=0, startY=0,
    xPad=0, yPad=0, shrink=1, blankSpace=blankSpace)


# Dimensions of a single char
charHeight, charWidth = cropped[0].shape
dim = charWidth * charHeight

charset = AnnoyIndex(dim, dist)
# Either build the model or load it
buildModel(dim, cropped)
charset.load('charset.ann')

# Resize photo to 10 characters wide
rphoto, targetPadding = resizePhoto(target, rowLength, (charWidth, charHeight), (xChange, yChange))
levelAdjustedChars = levelAdjustChars(cropped)

t = genTypable(rphoto, charset, levelAdjustedChars, kBest)
m = genMockup(t, cropped, (target.shape[1], target.shape[0]), targetPadding)

mockupFn = f'mockup/mockup_{rowLength}_best{kBest}_{ditherMode}.png'
print("writing file:")
print(mockupFn)
cv2.imwrite(mockupFn, m)