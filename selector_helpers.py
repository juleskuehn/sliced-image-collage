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

def buildModel(dim, slices, distMetric, trees=100):
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


def ok(idx, constraints, idxToConstraint):
    chars = idxToConstraint[idx]
    print(chars)
    for i in range(len(constraints)):
        if constraints[i] and chars[i] != constraints[i]:
            # print(chars)
            # print("failed with constraints:")
            # print(constraints)
            # print("---")
            return False
    return True


# Selects source slice (index) with lowest MSE vs target slice
def getSimilar(v, angularNN, euclideanNN, shapeliness, errCorrect, levelAdjustedChars, constraints, idxToConstraint):
    # Before applying dither, find the best matches for shape
    n = len(idxToConstraint) - 1
    aIndices, aScores = angularNN.get_nns_by_vector(np.ndarray.flatten(v), n, include_distances=True)
    eIndices, eScores = euclideanNN.get_nns_by_vector(np.ndarray.flatten(v), n, include_distances=True)

    
    maxAngular = np.max(aScores)
    minAngular = np.min(aScores)
    maxEuclidean = np.max(eScores)
    minEuclidean = np.min(eScores)
    
    aScores = (aScores - minAngular) / (maxAngular - minAngular)
    eScores = (eScores - minEuclidean) / (maxEuclidean - minEuclidean)
    aScores *= shapeliness
    eScores *= (1 - shapeliness)

    aDict = dict(zip(aIndices, aScores))
    eDict = dict(zip(eIndices, eScores))

    cDict = {key: aDict.get(key, 0) + eDict.get(key, 0)
          for key in set(aDict) | set(eDict) if ok(key, constraints, idxToConstraint)}

    return min(cDict.items(), key=operator.itemgetter(1))[0]


    # bestIdx = None
    # minScore = inf
    # for i in bestEuclidean:
    #     score = compare_mse(v, levelAdjustedChars[i])
    #     # score = abs(np.average(v) - np.average(levelAdjustedChars[i]))
    #     if score < minScore:
    #         minScore = score
    #         bestIdx = i
    # return values.index(min(values))


# Returns 2d array: indices of chosen source slices to best approximate target image
def genTypable(photo, charShape, angularNN, euclideanNN, kBest, levelAdjustedChars, idxToConstraint):
    height, width = photo.shape
    charHeight, charWidth = charShape
    typable = np.full((height//charHeight+2, width//charWidth+2), None, dtype=object)
    ditherMap = np.zeros((height//charHeight+2, width//charWidth+2), dtype=object)
    err = 0 # For dither
    for y in range(typable.shape[0] - 2):
        for x in range(typable.shape[1] - 2):
            startY = y*charHeight
            startX = x*charWidth
            endY = (y+1)*charHeight
            endX = (x+1)*charWidth
            v = photo[startY:endY, startX:endX].copy()
            constraints = getConstraints(typable, y+1, x+1, idxToConstraint)
            print(y, x, constraints)
            chosenIdx = getSimilar(
                v, angularNN, euclideanNN, kBest, ditherMap[y+1, x+1], levelAdjustedChars, constraints, idxToConstraint)
            typable[y+1, x+1] = chosenIdx
            # err = np.average(levelAdjustedChars[chosenIdx]) - np.average(v)
            # ditherMap = ditherFS(y+1, x+1, err, ditherMap)
            # print(ditherMap)
    return typable[1:-1, 1:-1]

def getConstraints(typable, y, x, idxToConstraint):
    TL = (-1, -1)
    TC = (-1, 0)
    TR = (-1, 1)
    L = (0, -1)
    R = (0, 1)
    BL = (1, -1)
    BC = (1, 0)
    BR = (1, 1) 
    TLc = idxToConstraint[typable[y+TL[0], x+TL[1]]]
    TCc = idxToConstraint[typable[y+TC[0], x+TC[1]]]
    TRc = idxToConstraint[typable[y+TR[0], x+TR[1]]]
    Lc = idxToConstraint[typable[y+L[0], x+L[1]]]
    Rc = idxToConstraint[typable[y+R[0], x+R[1]]]
    BLc = idxToConstraint[typable[y+BL[0], x+BL[1]]]
    BCc = idxToConstraint[typable[y+BC[0], x+BC[1]]]
    BRc = idxToConstraint[typable[y+BR[0], x+BR[1]]]

    # (Top left, Top right, Bottom left, Bottom right)
    constraints = [None, None, None, None]
    # BR character of TL must be TL char in this slice
    if not constraints[0]:
        constraints[0] = TLc[3]

    # # BL character of TC must be TL char in this slice
    # if not constraints[0]:
    #     constraints[0] = TCc[2]
    # # BR char of TC must be TR of this slice
    # if not constraints[1]:
    #     constraints[1] = TCc[3]

    # BL character of TR must be TR char in this slice
    if not constraints[1]:
        constraints[1] = TRc[2]

    # # TR char of L must be TL char in this slice
    # if not constraints[0]:
    #     constraints[0] = Lc[1]
    # BR char of L must be BL char in this slice
    if not constraints[2]:
        constraints[2] = Lc[3]
    
    # TL char of R must be TR char in this slice
    # if not constraints[1]:
    #     constraints[1] = Rc[0]
    # BL char of R must be BR char in this slice
    # if not constraints[3]:
    #     constraints[3] = Rc[2]

    # TR char of BL must be BL in this slice
    # if not constraints[2]:
    #     constraints[2] = BLc[1]
    
    # TL character of BC must be BL char in this slice
    # if not constraints[2]:
    #     constraints[2] = BCc[0]
    # TR char of BC must be BR of this slice
    # if not constraints[3]:
    #     constraints[3] = BCc[1]

    # TL character of BR must be BR char in this slice
    # if not constraints[3]:
    #     constraints[3] = TRc[0]

    print(constraints)
    return constraints



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