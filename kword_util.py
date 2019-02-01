import numpy as np
import matplotlib.pyplot as plt
import math
import random
from timeit import Timer
from annoy import AnnoyIndex
from matplotlib.image import imread, imsave
from PIL import Image
import cv2
from math import inf, floor, ceil
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr
import operator


# Resizes targetImg to be a multiple of character width
# Scales height to correct for change in proportion
# Pads height to be a multiple of character height
def resizeTarget(im, rowLength, charShape, charChange):
    charHeight, charWidth = charShape
    xChange, yChange = charChange
    inHeight, inWidth = im.shape
    outWidth = rowLength * charWidth
    outHeight = round((outWidth / inWidth) * inHeight * (xChange/yChange))
    im = cv2.resize(im, dsize=(outWidth, outHeight),
                    interpolation=cv2.INTER_AREA )
    print("resized target has shape", im.shape)
    # Pad outHeight so that it aligns with a character boundary
    if outHeight % charHeight != 0:
        newHeight = (outHeight//charHeight + 1) * charHeight
        blank = np.full((newHeight, outWidth), 255, dtype='uint8')
        blank[0:outHeight] = im
        blank[outHeight:] = np.tile(im[-1],(newHeight-outHeight,1))
        im = blank
        print("target padded to", im.shape)
    else:
        newHeight = inHeight
    return im, newHeight-outHeight


def brightenTarget(im, comboSet):
    minCombo = np.min([np.average(combo.img) for combo in comboSet.byIdx])
    print(minCombo)
    imBlack = 0 # Could test target image for this, or just leave as 0
    diff = minCombo - imBlack
    return im * (255-diff)/255 + diff


# Returns a mockup image, with the same size as the target image
def genMockup(comboGrid, comboSet, targetShape, targetPadding):
    gridShape = comboGrid.grid.shape
    comboShape = comboSet.byIdx[0].img.shape
    # Generate output image
    mockup = np.zeros((gridShape[0]*comboShape[0],
                       gridShape[1]*comboShape[1]), dtype='uint8')

    # print(comboGrid)
    for i, row in enumerate(comboGrid.grid):
        startY = i * comboShape[0]
        endY = (i + 1) * comboShape[0]
        for j, combo in enumerate(row):
            startX = j * comboShape[1]
            endX = (j + 1) * comboShape[1]
            mockup[startY:endY,startX:endX] = comboSet.byCombo[combo].img


    # Crop and resize mockup to match target image
    if targetPadding > 0:
        mockup = mockup[:-targetPadding, :]
        print("cropped to", mockup.shape)
    # return mockup
    resized = cv2.resize(mockup, dsize=(targetShape[1],targetShape[0]), interpolation=cv2.INTER_AREA)
    print("mockup has shape", resized.shape)
    return resized



# Returns ([cropped images], [padded images], (cropPosX, cropPosY))
# Cropped images are used for comparison (selection)
# Padded images can be used for reconstruction (mockup) but are not strictly necessary
def chop_charset(fn='hermes.png', numX=79, numY=7, startX=0, startY=0, xPad=0, yPad=0, shrink=1, blankSpace=True):
    """
    The trick is that each quadrant needs to be integer-sized (unlikely this will occur naturally), while maintaining proportionality. So we do some resizing and keep track of the changes in proportion:

    1. Rotate/crop scanned image to align with character boundaries
    - . (period) character used for this purpose, bordering the charset:
    .........
    . A B C .
    . D E F .
    .........
    - image is cropped so that each surrounding period is cut in half
    This is done manually, before running this script.

    2. Resize UP cropped image (keeping proportion) to be evenly divisible by (number of characters in the X dimension * 2). The * 2 is because we will be chopping the characters into quadrants.
    3. Resize in the y dimension (losing proportion) to be evenly divisible by (number of characters in the Y dimension * 2). Save (resizedY / originalY) as propChange.

    4. Now the charset image can be evenly sliced into quadrants. The target image (ie. photograph of a face, etc) must be resized in the Y dimension by propChange before processing. Then, the output from processing (ie. the typed mockup) must also be resized by 1/propChange.

    The issue of characters overextending their bounds cannot be fully addressed without substantial complication. We can pad the images during chopping, and then find a crop window (character size before padding) that maintains the most information from the padded images, ie. the sum of the cropped information is lowest across the character set.
    """
    im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    # im = imread(fn)[:,:,0]*255
    print("charset has shape", im.shape)
    # numX = 80  # Number of columns
    # numY = 8  # Number of rows

    stepX = im.shape[1]/numX  # Slice width
    stepY = im.shape[0]/numY  # Slice height

    # Need to resize charset such that stepX and stepY are each multiples of 2
    # After this, we can shrink without loss of proportion
    # shrinkFactor = 2
    newStepX = ceil(stepX/shrink)
    newStepY = ceil(stepY/shrink)
    # Ensure multiple of 2
    if newStepX % 2 == 1:
        newStepX += 1
    if newStepY % 2 == 1:
        newStepY += 1

    xChange = stepX / newStepX
    yChange = stepY / newStepY

    im = cv2.resize(im, dsize=(newStepX * numX, newStepY * numY), interpolation=cv2.INTER_AREA)
    print(np.max(im))
    print("Actual character size", stepX, stepY)
    print("Resized char size", newStepX, newStepY)
    print("resized charset has shape", im.shape)

    # These need manual tuning per charset image
    startX = int(startX*newStepX)  # Crop left px
    startY = int(startY*newStepY)  # Crop top px

    # Hidden hyperparameter
    whiteThreshold = 0.99 # Don't save images that are virtually empty

    tiles = []

    for y in range(startY, im.shape[0], newStepY):
        for x in range(startX, im.shape[1], newStepX):
            if np.sum(im[y:y+newStepY, x:x+newStepX][:,:]) < (newStepX*newStepY*whiteThreshold*255):
                tiles.append(im[y-yPad:y+newStepY+yPad, x-xPad:x+newStepX+xPad][:,:])
    # Append blank tile
    if blankSpace:
        tiles.append(np.full((newStepY+yPad*2, newStepX+xPad*2), 255.0, dtype='uint8'))
        
    print(len(tiles), 'characters chopped.')

    a = np.array(tiles)

    maxCroppedOut = -inf
    maxCropXY = (0, 0) # Top left corner of crop window

    ySize, xSize = a[0].shape
    ySize -= yPad * 2 # Target crop
    xSize -= xPad * 2 # Target crop

    # Try all the crops and find the best one (the one with most white)
    for y in range(yPad * 2):
        for x in range(xPad * 2):
            croppedOut = np.sum(a) - np.sum(a[:,y:y+ySize,x:x+xSize])
            if croppedOut > maxCroppedOut:
                maxCroppedOut = croppedOut
                maxCropXY = (x, y)

    x, y = maxCropXY
    print('cropped at ', x, y)
    # np.save('cropped_chars.npy', a[:,y:y+ySize,x:x+xSize])

    return a[:,y:y+ySize,x:x+xSize], tiles, (x, y), (xChange, yChange)