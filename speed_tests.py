import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import cv2
from math import inf, ceil, floor

import timeit

# from selector_helpers import chop_charset

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
    # print("charset has shape", im.shape)
    # numX = 80  # Number of columns
    # numY = 8  # Number of rows

    stepX = im.shape[1]/numX  # Slice width
    stepY = im.shape[0]/numY  # Slice height

    # Need to resize charset such that stepX and stepY are each multiples of 2
    # Also shrinking while we're at it
    # shrinkFactor = 2
    newStepX = ceil(stepX/shrink)
    newStepY = ceil(stepY/shrink)

    xChange = stepX / newStepX
    yChange = stepY / newStepY

    im = cv2.resize(im, dsize=(newStepX * numX, newStepY * numY), interpolation=cv2.INTER_CUBIC)
    # print(np.max(im))
    # print("Actual character size", stepX, stepY)
    # print("Resized char size", newStepX, newStepY)
    # print("resized charset has shape", im.shape)

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
        
    # print(len(tiles), 'characters chopped.')

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
    # print('cropped at ', x, y)
    # np.save('cropped_chars.npy', a[:,y:y+ySize,x:x+xSize])

    return tiles

def buildModel():
    dim = slices.shape[1] * slices.shape[2]
    model = AnnoyIndex(dim, metric='euclidean')
    for i, slice in enumerate(slices):
        model.add_item(i, np.ndarray.flatten(slice))
    model.build(10)
    # model.save(distMetric+'.ann')
    return model

def getSimilarNN():
    a = model.get_nns_by_vector(np.random.rand(45), 10, include_distances=True)
    return a



slices = np.random.rand(100000,5,9)
model = buildModel()

# print('chopCharset x100:', timeit.timeit( chop_charset, number=100 ))
print('buildModel(100000) x1):', timeit.timeit( buildModel, number=1 ))
kBest = 100000
print('getSimilarNN(1000000) x10000):', timeit.timeit( getSimilarNN, number=10000 ))

idxs, scores = getSimilarNN()

def findMatchingIdx():
    for idx in idxs:
        if idx == -1:
            return

print('findMatchingIdx(100000) x10000):', timeit.timeit( findMatchingIdx, number=10000 ))

# [print(idxs[i], scores[i]) for i in range(len(idxs))]