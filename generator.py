import numpy as np
import operator
import timeit
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from skimage.measure import compare_ssim, compare_mse
from combo import ComboSet, Combo
from combo_grid import ComboGrid
from char import Char
from kword_util import genMockup, gammaCorrect

class Generator:

    # Assumes targetImg has already been resized and padded to match combo dimensions
    def __init__(self, targetImg, shrunkenTargetImg, charSet, shapeliness=0.5,
                        targetShape=None, targetPadding=None, shrunkenTargetPadding=None, dither=True):
        self.targetImg = targetImg
        print('t', targetImg.shape)
        self.shrunkenTargetImg = shrunkenTargetImg
        print('s', shrunkenTargetImg.shape)
        self.ditherImg = shrunkenTargetImg.copy()
        self.charSet = charSet
        self.comboSet = ComboSet()
        self.comboH, self.comboW = charSet.get(0).cropped.shape[0]//2, charSet.get(0).cropped.shape[1]//2
        self.shrunkenComboH, self.shrunkenComboW = charSet.get(0).shrunken.shape[0]//2, charSet.get(0).shrunken.shape[1]//2
        self.mockupRows = targetImg.shape[0] // self.comboH
        self.mockupCols = targetImg.shape[1] // self.comboW
        print('mockupRows', self.mockupRows, 'mockupCols', self.mockupCols)
        self.rows = shrunkenTargetImg.shape[0] // self.shrunkenComboH
        self.cols = shrunkenTargetImg.shape[1] // self.shrunkenComboW
        print('rows      ', self.rows,       'cols      ', self.cols)
        self.targetShape = targetShape or targetImg.shape
        self.mockupImg = np.full(targetImg.shape, 255, dtype='uint8')
        self.fixedMockupImg = np.full(targetImg.shape, 255, dtype='uint8')
        self.shrunkenMockupImg = np.full(shrunkenTargetImg.shape, 255, dtype='uint8')
        self.targetPadding = targetPadding or 0
        self.shrunkenTargetPadding = shrunkenTargetPadding or 0
        self.comboGrid = ComboGrid(self.rows, self.cols)
        self.compareMode = 'mse'
        self.numLayers = 0 # How many times has the image been typed
        self.overtype = 1 # How many times have all 4 layers been typed
        self.firstPass = True
        self.maxGamma = [0.6,0.75,0.9,1]
        self.stats = {
            'positionsVisited': 0,
            'comparisonsMade': 0
        }
        self.dither = dither
        self.boostK = 0


    def getSliceBounds(self, row, col, shrunken=False):
        h = self.shrunkenComboH if shrunken else self.comboH
        w = self.shrunkenComboW if shrunken else self.comboW
        startY = row * h
        startX = col * w
        endY = (row+2) * h
        endX = (col+2) * w
        return startX, startY, endX, endY


    def putBetter(self, row, col, k):
        self.stats['positionsVisited'] += 1
        bestMatch = self.getBestOfRandomK(row, col, k+self.boostK)
        if bestMatch:
            # print(self.comboGrid.get(row, col), bestMatch)
            changed = True
            self.comboGrid.put(row, col, bestMatch)
            if self.dither:
                startX, startY, endX, endY = self.getSliceBounds(row, col, shrunken=True)
                self.shrunkenMockupImg[startY:endY, startX:endX] = self.compositeAdj(row, col, shrunken=True)
            startX, startY, endX, endY = self.getSliceBounds(row, col, shrunken=False)
            if row < self.mockupRows-1 and col < self.mockupCols-1:
                self.mockupImg[startY:endY, startX:endX] = self.compositeAdj(row, col, shrunken=False)
        else:
            # print("already good")
            changed = False
            # if not self.dither:
            self.comboGrid.clean(row, col)
        if self.dither:
            self.ditherImg = self.applyDither(row, col)
        return changed


    def getBestOfRandomK(self, row, col, k=5, binned=False):
        # Score against temporary ditherImg created for this comparison
        def compare(row, col, ditherImg):
            startX, startY, endX, endY = self.getSliceBounds(row, col, shrunken=self.dither)
            if self.dither:
                targetSlice = ditherImg[startY:endY, startX:endX]
            else:
                targetSlice = self.targetImg[startY:endY, startX:endX]
            mockupSlice = self.compositeAdj(row, col, shrunken=self.dither)
            # print('t', targetSlice.shape, 'm', mockupSlice.shape)
            # brighten targetSlice and compare
            # avgErr = abs(np.average(targetSlice/self.maxGamma - mockupSlice))
            targetSlice = gammaCorrect(targetSlice, self.maxGamma)
            mse = np.sqrt(compare_mse(targetSlice, mockupSlice))
            # Another metric: quadrant differences
            h, w = mockupSlice.shape
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
            return mse

        ditherImg = self.ditherImg
        curScore = compare(row, col, ditherImg)
        # print('curScore:',curScore)
        if binned:
            chars = self.charSet.getSorted()
            # Always include space
            charIdx = [0]
            for i in range(k):
                startIdx = max(1, (len(chars)*i)//k)
                endIdx = (len(chars)*(i+1))//k
                charIdx.append(np.random.randint(startIdx, endIdx))
            chars = list(np.array(chars)[charIdx])
        else:
            chars = self.charSet.getAll()
            # print(chars)
            # print(k)
            chars = list(np.random.choice(chars, k, replace=False))
            chars = chars + self.charSet.getAll()[:5]
        scores = {}
        origGrid = self.comboGrid.grid.copy()
        for char in chars:
            self.stats['comparisonsMade'] += 1
            self.comboGrid.put(row, col, char.id)
            if self.dither:
                ditherImg = self.applyDither(row, col)
            # Score the composite
            scores[char.id] = compare(row, col, ditherImg)

        self.comboGrid.grid = origGrid

        bestChoice = min(scores, key=scores.get)
        # Has to be 5% better
        betterRatio = 0.05
        print((curScore-scores[bestChoice])/curScore)
        better = scores[bestChoice] < curScore and (curScore-scores[bestChoice])/curScore > betterRatio
        return bestChoice if better else None


    # Return updated copy of the dither image based on current selection
    def applyDither(self, row, col, amount=0.3, mode='li'):
        # print("Begin dither")
        ditherImg = self.ditherImg.copy()
    
        startX, startY, endX, endY = self.getSliceBounds(row, col, shrunken=True)
        ditherDone = np.zeros(ditherImg.shape, dtype=np.bool)
        h, w = ditherImg.shape
        residual = 0

        if mode == 'li':
            # Li dither by pixel
            K = 2.6 # Hyperparam
            M = 7   # Mask size
            c = M//2

            # Calculate error between chosen combo and target subslice
            # Per pixel
            for row in range(startY, endY):
                for col in range(startX, endX):
                    # print(row, col)
                    actual = self.shrunkenMockupImg[row, col]
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
                            if (i, j) != (row, col) and not ditherDone[i, j]:
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
                    ditherImg[i, j] = self.shrunkenTargetImg[i, j]

        elif mode == 'fs':
            for row in range(startY, endY):
                for col in range(startX, endX):
                    actual = self.shrunkenMockupImg[row, col]
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
    def compositeAdj(self, row, col, shrunken=False):
        
        def getIndices(cDict):
            t = cDict[0], cDict[1], cDict[2], cDict[3]
            # print(cDict)
            return t

        def getChars(cDict):
            return (
                self.charSet.getByID(cDict[0]),
                self.charSet.getByID(cDict[1]),
                self.charSet.getByID(cDict[2]),
                self.charSet.getByID(cDict[3])
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
            idx = getIndices(self.comboGrid.get(aRow, aCol))
            combo = self.comboSet.getCombo(*idx)
            if not combo:
                # Combo not found
                chars = getChars(self.comboGrid.get(aRow, aCol))
                combo = self.comboSet.genCombo(*chars)
            qs[posID] = combo

        # Stitch quadrants together
        startX, startY, endX, endY = self.getSliceBounds(row, col, shrunken=False)
        img = self.fixedMockupImg[startY:endY, startX:endX] / 255
        img[:img.shape[0]//2, :img.shape[1]//2] *= qs[0].img
        img[:img.shape[0]//2, img.shape[1]//2:] *= qs[1].img 
        img[img.shape[0]//2:, :img.shape[1]//2] *= qs[2].img
        img[img.shape[0]//2:, img.shape[1]//2:] *= qs[3].img
        if shrunken:
            img = cv2.resize(img, dsize=(self.shrunkenComboW*2, self.shrunkenComboH*2), interpolation=cv2.INTER_AREA)
        return np.array(img * 255, dtype='uint8')


    def generateLayers(self, compareModes=['m'], numAdjustPasses=0,
                        show=True, mockupFn='mp_untitled', gamma=1,
                        randomInit=False, randomOrder=False):

        def dirtyLinearPositions(randomize=False, zigzag=True):
            positions = []
            for layerID in [0, 3, 1, 2]:
                startIdx = len(positions)
                r2l = False
                startRow = 0
                startCol = 0
                endRow = self.rows - 1
                endCol = self.cols - 1
                if layerID in [2, 3]:
                    startRow = 1
                    r2l = True
                if layerID in [1, 3]:
                    startCol = 1
                for row in range(startRow, endRow, 2):
                    for col in range(startCol, endCol, 2):
                        if self.comboGrid.isDirty(row,col):
                            positions.append((row, col))
                if r2l and zigzag:
                    positions[startIdx:len(positions)] = positions[len(positions)-1:startIdx-1:-1]
                positions.append(None)
            if randomize:
                np.random.shuffle(positions)
            # print(positions)
            # exit()
            return positions

        def setupFig():
            fig, ax = plt.subplots(1, 2)
            return fig, ax

        modeDict = {
            'm':'mse',
            'd':'dither'
        }
        compareModes = [modeDict[c] for c in compareModes]
        self.maxGamma = gamma
        self.compareMode = compareModes.pop(0)
        if self.compareMode == 'dither':
            self.dither = True
        else:
            self.dither = False
        self.positions = dirtyLinearPositions(randomize=randomOrder)

        # Initialize randomly if desired
        numChars = len(self.charSet.getAll())
        if randomInit:
            while len(self.positions) > 0:
                pos = self.positions.pop(0)
                if pos is None:
                    continue
                row, col = pos
                startX, startY, endX, endY = self.getSliceBounds(row, col)
                self.comboGrid.put(row, col, np.random.randint(1, numChars+1))
                self.mockupImg[startY:endY, startX:endX] = self.compositeAdj(row, col)

        fig, ax = setupFig()
        self.adjustPass = 0
        printEvery = 50

        def animate(frame):
            if frame % printEvery == 0:
                print(self.stats['positionsVisited'], 'positions visited')
                print(self.stats['comparisonsMade'], 'comparisons made')
                print(len(dirtyLinearPositions()), 'dirty positions remaining')
                print('---')
            if len(self.positions) == 0:
                print("Finished pass")
                # self.boostK += 2
                # self.comboGrid.printDirty()
                # print(self.comboGrid)
                self.positions += dirtyLinearPositions(randomize=randomOrder)
                # print("dirty:", len(self.positions))
                if len(self.positions) < 10: 
                    # There were no more dirty positions: next pass!
                    self.fixedMockupImg = self.mockupImg.copy()
                    self.compareMode = compareModes.pop(0)
                    self.dither = self.compareMode == 'dither'
                    self.comboGrid = ComboGrid(self.rows, self.cols)
                    self.positions = dirtyLinearPositions(randomize=randomOrder)
                    # Need to reset combos because running out of memory
                    self.comboSet = ComboSet()

            pos = self.positions.pop(0)
            if pos is None:
                # self.ditherImg = self.targetImg.copy()
                return
            row, col = pos
            # if self.putBestAdj(row, col):
            if self.putBetter(row, col, 5 if self.dither else 25): # best of k random
            # if self.putBetter(row, col, 1): # first random better
                ax[0].clear()
                ax[0].imshow(self.mockupImg, cmap='gray')
            ax[1].clear()
            ax[1].imshow(self.ditherImg if self.dither else self.targetImg, cmap='gray')

        # numFrames = (len(self.positions)-4)*(len(compareModes)+1+numAdjustPasses)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='Jules Kuehn'), bitrate=1800)
        ani = animation.FuncAnimation(fig, animate, repeat=False, frames=10000, interval=1)
        if show:
            plt.show()
        else:
            ani.save(mockupFn+'.mp4', writer=writer)

        print("Finished!")
        print(self.stats['positionsVisited'], 'positions visited')
        print(self.stats['comparisonsMade'], 'comparisons made')

        return self.comboGrid