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

"""     # print('''Usage: python dither_selector.py sourceFn targetFn slicesX slicesY rowLength bestK ditherMode [noblank]
    #     sourceFn, targetFn: file paths
    #     slicesX, slicesY: positive integers
    #     rowLength: best set to same as slicesX, for testing reconstruction of the same sourceFn vs targetFn
    #     bestK: how many nearest neighbours are given to the MSE evaluator
    #     ditherMode: one of ["fs", "li", "priority", "none"]
    #     noblank: this word will override the default behaviour, which is to keep a blank space in the charset.
    #     ''') """

sourceFn = args[1]
targetFn = args[2]
slicesX = int(args[3])
slicesY = int(args[4])
rowLength = int(args[5])
c = int(args[6])
shrinkX = int(args[7])
shrinkY = int(args[8])
dither = 'dither' in args
preview = 'preview' in args

targetImg = cv2.imread(targetFn, cv2.IMREAD_GRAYSCALE)
print("target photo has shape", targetImg.shape)

# Get character set and related info
xPad = 4
yPad = 4
cropped, padded, (xCropPos, yCropPos), (xChange, yChange) = chop_charset(
    fn=sourceFn, numX=slicesX, numY=slicesY, startX=0, startY=0,
    xPad=xPad, yPad=yPad, shrinkX=shrinkX, shrinkY=shrinkY, blankSpace=True)

""" sortedCropIdx = np.argsort([np.average(char) for char in cropped])[::-1]
# print(sortedCropIdx)
m = len(sortedCropIdx)//2 - c//2
chooseThese = list(sortedCropIdx[:c])+list(sortedCropIdx[m:m+c])+list(sortedCropIdx[-c:])+[6, 8, 19, 75]
bestChars = cropped[chooseThese]
# randomCharIdx = list(np.random.choice(len(cropped)-1, numChars)) """

# # Save characters
import os
d = os.getcwd() + '\\chars'
filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
for f in filesToRemove:
    os.remove(f) 
for i, char in enumerate(padded):
    cv2.imwrite('chars/padded_'+str(i)+'.png', char)

# Create Char objects from [padded]
cropSettings = {
    'xPad': xPad,
    'yPad': yPad,
    'xCropPos': xCropPos,
    'yCropPos': yCropPos,
    'shrinkX': shrinkX,
    'shrinkY': shrinkY
}
charset = CharSet(padded, cropSettings)
[print(char) for char in charset.getSorted()]


# Create combos of darkest characters (to find a black level)
# Store combos in a sparse 4d array (defaultdict), more to be added as selection proceeds
""" 

comboSet = ComboSet(CharSet(bestChars))

# cv2.imwrite('combo_first.png', comboSet.byIdx[0].img)
# cv2.imwrite('combo_last.png', comboSet.byIdx[-1].img)

# for combo in comboSet.byIdx:
#     cv2.imwrite('combos/combo_'+str(combo.idx)+'.png', combo.img)

# Resize target photo to rowLength * charWidth and pad to next multiple of charHeight
resizedTarget, targetPadding = resizeTarget(targetImg, rowLength,  cropped[0].shape, (xChange, yChange))

# cv2.imwrite('sobel.png', cv2.Laplacian(resizedTarget,cv2.CV_64F))
brightenAmount = 1
resizedTarget = brightenTarget(resizedTarget, comboSet, brightenAmount)
print(resizedTarget.dtype)

generator = Generator(resizedTarget, comboSet, targetShape=targetImg.shape, targetPadding=targetPadding, dither=dither)
cv2.imwrite('lapTest.png', generator.testPriorityOrder())


filledComboGrid = generator.generatePriorityOrder(preview=preview)

# # Generate  mockup (reconstruction of target in terms of source)
m = genMockup(filledComboGrid, comboSet, targetImg.shape, targetPadding)

mockupFn = f"mockup/mockup_{sourceFn.split('.')[-2][1:]}_{targetFn.split('.')[-2][1:]}_{rowLength}w_c{c}_shrink{shrink}_{dither}"
print("writing file:")
print(mockupFn)

# cv2.imwrite(mockupFn+'c.png', cv2.addWeighted(m,0.5,targetImg,0.5,0))
cv2.imwrite(mockupFn+'.png', m)

ax1 = plt.subplot(111)
#create image plot
im1 = ax1.imshow(m,cmap='gray')
plt.show() """