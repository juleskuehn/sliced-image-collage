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
shapeliness = float(args[6])

target = cv2.imread(targetImg, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", target.shape)

# Get character set and related info
cropped, padded, (xPad, yPad), (xChange, yChange) = chop_charset(
    fn=sourceImg, numX=slicesX, numY=slicesY, startX=0, startY=0,
    xPad=0, yPad=0, shrink=1, blankSpace=True)

"""     # 8: 0
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
    # bestChars = cropped[[0,6,8,9,10,16,19,21,30,33,35,36,37,38,41,42,43,44,45,46,47,48,85,86,-1]]
    # bestChars = cropped[[0,6,8,9,10,16,19,21,30,33,35,36,37,38,41,42,43,44,45,46,47,48,49,50,51,52,65,71,75,76,78,83,84,85,86,-1]] """
bestChars = cropped[[-1, 65]] # Blank space always first

comboSet = ComboSet(len(bestChars), charset=CharSet(bestChars))

# cv2.imwrite('combo_first.png', comboSet.byIdx[0].img)
# cv2.imwrite('combo_last.png', comboSet.byIdx[-1].img)

# for combo in comboSet.byIdx:
#     cv2.imwrite('combo_'+str(combo.idx)+'.png', combo.img)

# Resize target photo to rowLength * charWidth and pad to next multiple of charHeight
resizedTarget, targetPadding = resizeTarget(target, rowLength,  comboSet.byIdx[0].img.shape, (xChange, yChange))

generator = Generator(resizedTarget, comboSet, shapeliness=shapeliness)
filledComboGrid = generator.generateLinearly()

# # Generate  mockup (reconstruction of target in terms of source)
m = genMockup(filledComboGrid, comboSet, target.shape, targetPadding)

mockupFn = f"mockup/mockup_{sourceImg.split('.')[-2][1:]}_{targetImg.split('.')[-2][1:]}_{rowLength}w_shape{shapeliness}.png"
print("writing file:")
print(mockupFn)
cv2.imwrite(mockupFn, m)