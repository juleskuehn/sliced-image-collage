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

def buildModel(dim, slices, distMetric, trees=10):
    model = AnnoyIndex(dim, metric=distMetric)
    for i, char in enumerate(slices):
        model.add_item(i, np.ndarray.flatten(char))
    model.build(trees)
    model.save(distMetric+'.ann')
    return model


# Resizes targetImg to be a multiple of character width
# Scales height to correct for change in proportion
# Pads height to be a multiple of character height
def resizePhoto(im, rowLength, charShape, charChange):
    charWidth, charHeight = charShape
    xChange, yChange = charChange
    inHeight, inWidth = im.shape
    outWidth = rowLength * charWidth
    outHeight = round((outWidth / inWidth) * inHeight * (xChange/yChange))
    im = cv2.resize(im, dsize=(outWidth, outHeight),
                    interpolation=cv2.INTER_CUBIC)
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


# Scales black/white levels of charset
# Uses global variable "cropped" for char levels
def levelAdjustChars(cropped):
    return cropped


# Selects source slice (index) with lowest MSE vs target slice
def getSimilar(v, angularNN, euclideanNN, kBest, errCorrect, levelAdjustedChars):
    # Before applying dither, find the best matches for shape
    bestAngular = angularNN.get_nns_by_vector(np.ndarray.flatten(v), kBest, include_distances=False)
    bestEuclidean = euclideanNN.get_nns_by_vector(np.ndarray.flatten(v), kBest, include_distances=False)
    bestIdx = None
    minScore = inf
    for i in bestEuclidean:
        score = compare_mse(v, levelAdjustedChars[i])
        # score = abs(np.average(v) - np.average(levelAdjustedChars[i]))
        if score < minScore:
            minScore = score
            bestIdx = i
    return bestIdx


# Returns 2d array: indices of chosen source slices to best approximate target image
def genTypable(photo, charShape, angularNN, euclideanNN, kBest, levelAdjustedChars):
    height, width = photo.shape
    charHeight, charWidth = charShape
    typable = np.zeros((height//charHeight, width//charWidth), dtype=object)
    ditherMap = np.zeros((height//charHeight+2, width//charWidth+2), dtype=object)
    err = 0 # For dither
    for y in range(typable.shape[0]):
        for x in range(typable.shape[1]):
            startY = y*charHeight
            startX = x*charWidth
            endY = (y+1)*charHeight
            endX = (x+1)*charWidth
            v = photo[startY:endY, startX:endX].copy()
            chosenIdx = getSimilar(
                v, angularNN, euclideanNN, kBest, ditherMap[y+1, x+1], levelAdjustedChars)
            typable[y, x] = chosenIdx
            # err = np.average(levelAdjustedChars[chosenIdx]) - np.average(v)
            # ditherMap = ditherFS(y+1, x+1, err, ditherMap)
            # print(ditherMap)
    return typable


def ditherFS(y, x, err, ditherMap):
    # ditherMap is padded by 1 in both dimensions
    ditherMap[y, x+1] += err * 7/16
    ditherMap[y+1, x-1] += err * 3/16
    ditherMap[y+1, x] += err * 5/16
    ditherMap[y+1, x+1] += err * 1/16

    return ditherMap


# Returns a mockup image, with the same size as the target image
def genMockup(typable, cropped, targetShape, targetPadding):
    tHeight, tWidth = typable.shape
    charHeight, charWidth = cropped[0].shape
    # Generate output image
    mockup = np.zeros((tHeight*charHeight, tWidth*charWidth), dtype='uint8')

    for y in range(tHeight):
        for x in range(tWidth):
            startY = y*charHeight
            startX = x*charWidth
            endY = (y+1)*charHeight
            endX = (x+1)*charWidth
            mockup[startY:endY, startX:endX] = cropped[typable[y, x]]

    # Crop and resize mockup to match target image
    if targetPadding > 0:
        mockup = mockup[:-targetPadding, :]
        print("cropped to", mockup.shape)
    resized = cv2.resize(mockup, dsize=targetShape, interpolation=cv2.INTER_CUBIC)
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
    # Also shrinking while we're at it
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

    im = cv2.resize(im, dsize=(newStepX * numX, newStepY * numY), interpolation=cv2.INTER_CUBIC)
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