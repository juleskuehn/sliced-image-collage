import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_mse
from kword_utils import gammaCorrect
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


def putBetter(generator, row, col, k):
    generator.stats['positionsVisited'] += 1
    bestMatch = getBestOfRandomK(generator, row, col, k+generator.boostK)
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


def getBestOfRandomK(generator, row, col, k=5, binned=False):
    # Score against temporary ditherImg created for this comparison
    def compare(row, col, ditherImg):
        
        if generator.dither:
            startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=True)
            ditherSlice = ditherImg[startY:endY, startX:endX]
            shrunkenMockupSlice = compositeAdj(generator, row, col, shrunken=True)
            # print('ditherSlice', ditherSlice.shape)
            # print('shrunkMock', shrunkenMockupSlice.shape)
        
        startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
        targetSlice = generator.targetImg[startY:endY, startX:endX]
        mockupSlice = compositeAdj(generator, row, col, shrunken=False)
        # print('targetSlice', targetSlice.shape)
        # print('mockupSlice', mockupSlice.shape)

        # avgErr = abs(np.average(targetSlice/generator.maxGamma - mockupSlice))
        score = 0
        if generator.compareMode in ['ssim']:
            # targetSlice = gammaCorrect(targetSlice, generator.gamma)
            score = -1 * compare_ssim(targetSlice, mockupSlice) + 1
            if generator.dither:
                ditherSlice = gammaCorrect(ditherSlice, generator.gamma)
                score *= np.sqrt(compare_mse(ditherSlice, shrunkenMockupSlice)) / 255
        elif generator.compareMode in ['mse', 'dither']:
            targetSlice = gammaCorrect(targetSlice, generator.gamma)
            score = np.sqrt(compare_mse(targetSlice, mockupSlice)) / 255
            if generator.dither:
                ditherSlice = gammaCorrect(ditherSlice, generator.gamma)
                score *= np.sqrt(compare_mse(ditherSlice, shrunkenMockupSlice)) / 255
        elif generator.compareMode in ['blend']:
            score = -1 * compare_ssim(targetSlice, mockupSlice) + 1
            # print('ssim score:', score)
            targetSlice = gammaCorrect(targetSlice, generator.gamma)
            score *= np.sqrt(compare_mse(targetSlice, mockupSlice)) / 255
            # print('combined score:', score)
        # Another metric: quadrant differences
        # h, w = mockupSlice.shape
        # mockupTLavg = np.average(mockupSlice[:h//2, :w//2])
        # mockupTRavg = np.average(mockupSlice[:h//2, w//2:])
        # mockupBLavg = np.average(mockupSlice[h//2:, :w//2])
        # mockupBRavg = np.average(mockupSlice[h//2:, w//2:])
        # targetTLavg = np.average(targetSlice[:h//2, :w//2])
        # targetTRavg = np.average(targetSlice[:h//2, w//2:])
        # targetBLavg = np.average(targetSlice[h//2:, :w//2])
        # targetBRavg = np.average(targetSlice[h//2:, w//2:])
        # quadrantErr = (
        #     abs(mockupTLavg-targetTLavg) +
        #     abs(mockupBLavg-targetBLavg) +
        #     abs(mockupBRavg-targetBRavg) +
        #     abs(mockupTRavg-targetTRavg)
        #     )/4
        return score

    ditherImg = generator.ditherImg
    curScore = compare(row, col, ditherImg)
    # print('curScore:',curScore)
    # if binned:
    #     chars = generator.charSet.getSorted()
    #     # Always include space
    #     charIdx = [0]
    #     for i in range(k):
    #         startIdx = max(1, (len(chars)*i)//k)
    #         endIdx = (len(chars)*(i+1))//k
    #         charIdx.append(np.random.randint(startIdx, endIdx))
    #     chars = list(np.array(chars)[charIdx])
    # elif generator.dither:
    #     chars = generator.charSet.getSorted()[:10] # brightest (m) chars
    #     # chars += list(np.array(generator.charSet.getSorted())[[21,22,40,62,77,87]]) # add o, O, 0, 8, #, @
    # else:
    #     chars = generator.charSet.getSorted()[10:] # all but brightest 10
    #     # print(chars)
    #     # print(k)
    #     chars = list(np.random.choice(chars, k, replace=False)) # choose k
    #     chars = chars + generator.charSet.getSorted()[:10] # add brightest 10
    chars = generator.charSet.getSorted()
    scores = {}
    origGrid = generator.comboGrid.grid.copy()
    for char in chars:
        generator.stats['comparisonsMade'] += 1
        generator.comboGrid.put(row, col, char.id)
        if generator.dither:
            ditherImg = applyDither(generator, row, col)
        # Score the composite
        scores[char.id] = compare(row, col, ditherImg)

    generator.comboGrid.grid = origGrid

    bestChoice = min(scores, key=scores.get)
    # Has to be some amount better
    betterRatio = 0
    # print((curScore-scores[bestChoice])/curScore)
    better = scores[bestChoice] < curScore and (curScore-scores[bestChoice])/curScore > betterRatio
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
