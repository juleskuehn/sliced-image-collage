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
from selector import Selector
from generator import Generator
from kword_util import *

args = sys.argv

"""     # print('''Usage: python dither_selector.py sourceImg targetImg slicesX slicesY rowLength bestK ditherMode [noblank]
    #     sourceImg, targetImg: file paths
    #     slicesX, slicesY: positive integers
    #     rowLength: best set to same as slicesX, for testing reconstruction of the same sourceImg vs targetImg
    #     bestK: how many nearest neighbours are given to the MSE evaluator
    #     ditherMode: one of ["fs", "li", "priority", "none"]
    #     noblank: this word will override the default behaviour, which is to keep a blank space in the charset.
    #     ''') """

sourceImg = args[1]
targetImg = args[2]
slicesX = int(args[3])
slicesY = int(args[4])
rowLength = int(args[5])
c = int(args[6])
shrink = int(args[7])

target = cv2.imread(targetImg, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", target.shape)

# Get character set and related info
cropped, padded, (xPad, yPad), (xChange, yChange) = chop_charset(
    fn=sourceImg, numX=slicesX, numY=slicesY, startX=0, startY=0,
    xPad=0, yPad=0, shrink=shrink, blankSpace=True)


sortedCropIdx = np.argsort([np.average(char) for char in cropped])[::-1]
print(sortedCropIdx)
m = len(sortedCropIdx)//2 - c//2
chooseThese = list(sortedCropIdx[:c])+list(sortedCropIdx[m:m+c])+list(sortedCropIdx[-c:])
bestChars = cropped[chooseThese]
# randomCharIdx = list(np.random.choice(len(cropped)-1, numChars))
import os
d = os.getcwd() + '\\chars'
filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
for f in filesToRemove:
    os.remove(f) 
for i, char in enumerate(bestChars):
    cv2.imwrite('chars/char_'+str(i+1)+'.png', char)

comboSet = ComboSet(CharSet(bestChars))

# cv2.imwrite('combo_first.png', comboSet.byIdx[0].img)
# cv2.imwrite('combo_last.png', comboSet.byIdx[-1].img)

# for combo in comboSet.byIdx:
#     cv2.imwrite('combos/combo_'+str(combo.idx)+'.png', combo.img)

# Resize target photo to rowLength * charWidth and pad to next multiple of charHeight
resizedTarget, targetPadding = resizeTarget(target, rowLength,  comboSet.byIdx[0].img.shape, (xChange, yChange))

# cv2.imwrite('sobel.png', cv2.Laplacian(resizedTarget,cv2.CV_64F))
brightenAmount = 1
resizedTarget = brightenTarget(resizedTarget, comboSet, brightenAmount)

generator = Generator(resizedTarget, comboSet)
filledComboGrid = generator.generatePriorityOrder()

# # Generate  mockup (reconstruction of target in terms of source)
m = genMockup(filledComboGrid, comboSet, target.shape, targetPadding)

mockupFn = f"mockup/mockup_{sourceImg.split('.')[-2][1:]}_{targetImg.split('.')[-2][1:]}_{rowLength}w_c{c}_shrink{shrink}.png"
print("writing file:")
print(mockupFn)
cv2.imwrite(mockupFn, m)