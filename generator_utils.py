import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_mse, compare_psnr
# from kword_utils import gammaCorrect
from math import ceil, sqrt

# Functions related to the selection of characters

def getSliceBounds(generator, row, col, shrunken=False):
    h = generator.shrunkenComboH if shrunken else generator.comboH
    w = generator.shrunkenComboW if shrunken else generator.comboW
    startY = row * h
    startX = col * w
    endY = (row+2) * h
    endX = (col+2) * w
    return startX, startY, endX, endY


# Comparison with ANN is simple, so all code built into this function
# vs. putBetter which calls getBestOfRandomK() which calls compare()
def putAnn(generator, row, col, mode='blend'):
    startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
    targetSlice = generator.targetImg[startY:endY, startX:endX]
    if mode in ['angular', 'blend']:
        return
    if mode in ['euclidean', 'blend']:
        return
    if mode == 'blend':
        return

def putBetter(generator, row, col, k):
    generator.stats['positionsVisited'] += 1
    k = min(len(generator.charSet.getAll()) - 5, k*generator.boostK)
    bestMatch = getBestOfRandomK(generator, row, col, k)
    # bestMatch = getNextBetter(generator, row, col, mode='sorted')
    if bestMatch:
        # print(generator.comboGrid.get(row, col), bestMatch)
        changed = True
        generator.comboGrid.put(row, col, bestMatch, chosen=True)
        if generator.dither:
            startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=True)
            generator.shrunkenMockupImg[startY:endY, startX:endX] = compositeAdj(generator, row, col, shrunken=True)
        startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
        if row < generator.mockupRows-1 and col < generator.mockupCols-1:
            generator.mockupImg[startY:endY, startX:endX] = compositeAdj(generator, row, col, shrunken=False)
    else:
        # print("already good")
        changed = False
        # if not generator.dither:
        generator.comboGrid.clean(row, col)
    if generator.dither:
        generator.ditherImg = applyDither(generator, row, col)
    return changed


def putSimAnneal(generator, row, col):
    generator.stats['positionsVisited'] += 1
    bestMatch = getSimAnneal(generator, row, col)
    if bestMatch:
        changed = True
        generator.comboGrid.put(row, col, bestMatch, chosen=True)
        if generator.dither:
            startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=True)
            generator.shrunkenMockupImg[startY:endY, startX:endX] = compositeAdj(generator, row, col, shrunken=True)
        startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
        if row < generator.mockupRows-1 and col < generator.mockupCols-1:
            generator.mockupImg[startY:endY, startX:endX] = compositeAdj(generator, row, col, shrunken=False)
    else:
        # print("already good")
        changed = False
        # Don't clean with simulated annealing unless temperature == 0
        if generator.getTemp() <= generator.minTemp*1.5:
            generator.comboGrid.clean(row, col)
    return changed


def getSimAnneal(generator, row, col):
    # Get score of existing slice
    curScore = compare(generator, row, col)
    chars = generator.charSet.getSorted()[:]
    np.random.shuffle(chars)
    origGrid = generator.comboGrid.grid.copy()
    newChar = None
    for char in chars:
        generator.comboGrid.put(row, col, char.id)
        newScore = compare(generator, row, col)
        # Note that delta is reversed because we are looking for a minima
        delta = curScore - newScore
        # if delta < 0:
            # print(np.exp(delta / generator.getTemp())*100)
        if delta > 0 or np.exp(delta / generator.getTemp()) > np.random.rand():
            newChar = char.id
            break
    generator.comboGrid.grid = origGrid
    return newChar


def initRandomPositions(generator):
    #Initialize randomly if desired
    numChars = len(generator.charSet.getAll())
    while len(generator.positions) > 0:
        pos = generator.positions.pop(0)
        if pos is None:
            continue
        row, col = pos
        startX, startY, endX, endY = getSliceBounds(generator, row, col)
        generator.comboGrid.put(row, col, np.random.randint(1, numChars+1))
        generator.mockupImg[startY:endY, startX:endX] = compositeAdj(generator, row, col)


def compare(generator, row, col, ditherImg=None):
    generator.stats['comparisonsMade'] += 1
    if generator.dither:
        startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=True)
        ditherSlice = ditherImg[startY:endY, startX:endX]
        shrunkenMockupSlice = compositeAdj(generator, row, col, shrunken=True)
    
    startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
    targetSlice = generator.targetImg[startY:endY, startX:endX]
    mockupSlice = compositeAdj(generator, row, col, shrunken=False)
    # avgErr = abs(np.average(targetSlice/generator.maxGamma - mockupSlice))
    score = 0
    if generator.compareMode in ['ssim']:
        # targetSlice = gammaCorrect(targetSlice, generator.gamma)
        score = -1 * compare_ssim(targetSlice, mockupSlice) + 1
        if generator.dither:
            # ditherSlice = gammaCorrect(ditherSlice, generator.gamma)
            score *= np.sqrt(compare_mse(ditherSlice, shrunkenMockupSlice)) / 255
    elif generator.compareMode in ['mse', 'dither']:
        # targetSlice = gammaCorrect(targetSlice, generator.gamma)
        score = compare_mse(targetSlice, mockupSlice)
        if generator.dither:
            # ditherSlice = gammaCorrect(ditherSlice, generator.gamma)
            score *= np.sqrt(compare_mse(ditherSlice, shrunkenMockupSlice)) / 255
    elif generator.compareMode in ['blend']:
        score = -1 * compare_ssim(targetSlice, mockupSlice) + 1
        # print('ssim score:', score)
        # targetSlice = gammaCorrect(targetSlice, generator.gamma)
        score *= np.sqrt(compare_mse(targetSlice, mockupSlice)) / 255
    elif generator.compareMode in ['armse']:
        # Asymmetric root mean squared error
        # TODO Broken!
        # targetSlice = gammaCorrect(targetSlice, generator.gamma)
        offset = 0
        score = np.sqrt(np.average(np.power(np.array(
            targetSlice - mockupSlice + offset,
            dtype='int16'), 2))) / 255
    # print(score)
    return score


def getNextBetter(generator, row, col, mode='sorted'):
    curScore = compare(generator, row, col)
    chars = generator.charSet.getSorted() if mode == 'sorted' else generator.charSet.getChars()
    if mode == 'random':
        chars = chars[:]
        np.random.shuffle(chars)
    origGrid = generator.comboGrid.grid.copy()
    betterChoice = None
    for char in chars:
        generator.comboGrid.put(row, col, char.id)
        if generator.dither:
            ditherImg = applyDither(generator, row, col)
        # Score the composite
        if compare(generator, row, col) < curScore:
            betterChoice = char.id
            break

    generator.comboGrid.grid = origGrid
    return betterChoice


def getBestOfRandomK(generator, row, col, k=5, binned=False):
    # Score against temporary ditherImg created for this comparison
    ditherImg = generator.ditherImg
    curScore = compare(generator, row, col, ditherImg)
    chars = generator.charSet.getSorted()[5:] # all but brightest 5
    chars = list(np.random.choice(chars, k, replace=False)) # choose k
    chars = chars + generator.charSet.getSorted()[:5] # add brightest 5
    # chars = generator.charSet.getSorted()
    scores = {}
    origGrid = generator.comboGrid.grid.copy()
    for char in chars:
        generator.comboGrid.put(row, col, char.id)
        if generator.dither:
            ditherImg = applyDither(generator, row, col)
        # Score the composite
        scores[char.id] = compare(generator, row, col, ditherImg=ditherImg)

    generator.comboGrid.grid = origGrid

    bestChoice = min(scores, key=scores.get)
    better = scores[bestChoice] < curScore
    return bestChoice if better else None


# Return updated copy of the dither image based on current selection
def applyDither(generator, row, col, amount=0.3, mode='li'):
    # print("Begin dither")
    ditherImg = generator.ditherImg.copy()

    startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=True)
    ditherDone = np.zeros(ditherImg.shape, dtype=np.bool)
    h, w = ditherImg.shape
    residual = 0

    if mode == 'li':
        # Li dither by pixel
        K = 2.6 # Hyperparam
        # Mask size
        M = 3
        # M = max(3, ceil((generator.shrunkenComboH+generator.shrunkenComboW)/4))
        if M % 2 == 0:
            M += 1
        # print(M)
        c = M//2

        # Calculate error between chosen combo and target subslice
        # Per pixel
        for row in range(startY, endY):
            for col in range(startX, endX):
                # print(row, col)
                actual = generator.shrunkenMockupImg[row, col]
                # desired = ditherImg[row, col] + residual
                # target = min(255, max(0, ditherImg[row, col] + residual))
                # residual = desired - target
                target = ditherImg[row, col] + residual
                error = (target - actual)*amount
                # print(error)
                # print(ditherSlice.shape, mockupSlice.shape)
                # Get adjacent pixels which aren't chosen already, checking bounds
                adjIdx = []
                for i in range(max(0,row-c), min(h, row+c+1)):
                    for j in range(max(0,col-c), min(w, col+c+1)):
                        # print(i,j)
                        # if (i, j) != (row, col) and not ditherDone[i, j]:
                        if (i, j) != (row, col):
                            adjIdx.append(
                                (i, j, np.linalg.norm(np.array([i,j])-np.array([row,col])))
                            )
                # print(adjIdx)
                weightIdx = []
                for i, j, dist in adjIdx:
                    adjVal = ditherImg[i, j]
                    # Darken slices which are already darker, and vice-versa
                    # Affect closer slices more
                    weight = (adjVal if error > 0 else 255 - adjVal) / (dist**K)
                    weightIdx.append((i, j, weight, adjVal))
                
                totalWeight = np.sum([weight for _, _, weight, _ in weightIdx])
                for i, j, weight, beforeVal in weightIdx:
                    # Normalize weights since not all slices will be adjustable
                    weight /= totalWeight
                    # Overall we want to reach this level with the slice:
                    # desiredVal = beforeVal + error*weight + residual
                    desiredVal = beforeVal + error*weight
                    # Apply corrections per pixel
                    correction = (desiredVal - beforeVal)
                    ditherImg[i, j] = min(255, max(0, ditherImg[i, j] + correction))
                    afterVal = ditherImg[i, j]
                    # residual = desiredVal - afterVal
                    # print(beforeVal, desiredVal - afterVal)
                
                ditherDone[row, col] = True
                ditherImg[i, j] = generator.shrunkenTargetImg[i, j]

    elif mode == 'fs':
        for row in range(startY, endY):
            for col in range(startX, endX):
                actual = generator.shrunkenMockupImg[row, col]
                target = ditherImg[row, col] + residual
                error = (target - actual)*amount
                if (col + 1 < w):
                    ditherImg[row, col+1] = min(255, max(0, ditherImg[row, col+1] + error * 7/16))
                # Dither Bottom Left
                if (row + 1 < h and col - 1 >= 0):
                    ditherImg[row+1, col-1] = min(255, max(0, ditherImg[row+1, col-1] + error * 3/16))
                # Dither Bottom
                if (row + 1 < w):
                    ditherImg[row+1, col] = min(255, max(0, ditherImg[row+1, col] + error * 5/16))
                # Dither BottomRight
                if (row + 1 < w and col + 1 < h):
                    ditherImg[row+1, col+1] = min(255, max(0, ditherImg[row+1, col+1] + error * 1/16))
    # print("end dither")
    return ditherImg
    

# Uses combos to store already composited "full" (all 4 layers)
# If combo not already generated, add it to comboSet.
# Returns mockupImg slice
def compositeAdj(generator, row, col, shrunken=False):
    
    def getIndices(cDict):
        t = cDict[0], cDict[1], cDict[2], cDict[3]
        # print(cDict)
        return t

    def getChars(cDict):
        return (
            generator.charSet.getByID(cDict[0]),
            generator.charSet.getByID(cDict[1]),
            generator.charSet.getByID(cDict[2]),
            generator.charSet.getByID(cDict[3])
        )
    
    qs = {} # Quadrants

    # TL , TR, BL, BR
    for posID in [0, 1, 2, 3]:
        aRow = row
        aCol = col
        if posID in [1, 3]:
            aCol += 1
        if posID in [2, 3]:
            aRow += 1
        idx = getIndices(generator.comboGrid.get(aRow, aCol))
        combo = generator.comboSet.getCombo(*idx)
        if not combo:
            # Combo not found
            chars = getChars(generator.comboGrid.get(aRow, aCol))
            combo = generator.comboSet.genCombo(*chars)
        qs[posID] = combo

    # Stitch quadrants together
    startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
    img = generator.fixedMockupImg[startY:endY, startX:endX] / 255
    img[:img.shape[0]//2, :img.shape[1]//2] *= qs[0].img
    img[:img.shape[0]//2, img.shape[1]//2:] *= qs[1].img 
    img[img.shape[0]//2:, :img.shape[1]//2] *= qs[2].img
    img[img.shape[0]//2:, img.shape[1]//2:] *= qs[3].img
    if shrunken:
        img = cv2.resize(img, dsize=(generator.shrunkenComboW*2, generator.shrunkenComboH*2), interpolation=cv2.INTER_AREA)
    return np.array(img * 255, dtype='uint8')

def evaluateMockup(generator):
    psnr = compare_psnr(generator.mockupImg, generator.targetImg)
    print("PSNR:", psnr)
    print("SSIM:", compare_ssim(generator.mockupImg, generator.targetImg))
    return psnr
