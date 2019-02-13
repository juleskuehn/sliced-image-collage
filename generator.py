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
    def __init__(self, targetImg, charSet, shapeliness=0.5,
                        targetShape=None, targetPadding=None, dither=False):
        self.targetImg = targetImg
        self.ditherImg = targetImg.copy()
        self.charSet = charSet
        self.comboSet = ComboSet()
        self.comboH, self.comboW = charSet.get(0).cropped.shape[0]//2, charSet.get(0).cropped.shape[1]//2
        self.rows = targetImg.shape[0] // self.comboH
        self.cols = targetImg.shape[1] // self.comboW
        self.targetShape = targetShape or targetImg.shape
        self.mockupImg = np.full(targetImg.shape, 255, dtype='uint8')
        self.targetPadding = targetPadding or 0
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
        self.dither = True

    def getSliceBounds(self, row, col):
        startY = row * self.comboH
        startX = col * self.comboW
        endY = (row+2) * self.comboH
        endX = (col+2) * self.comboW
        return startX, startY, endX, endY


    # Finding best character that is BR at (row, col)
    # If the character is already placed there, return False
    # Otherwise, place it (update comboGrid and mockup) and return True
    def putBestAdj(self, row, col):
        self.stats['positionsVisited'] += 1
        startX, startY, endX, endY = self.getSliceBounds(row, col)
        # Get ID of best match
        # If dither, this will call applyDither on every mockup before scoring
        bestMatch = self.getBestAdj(row, col)
        # print(bestMatch)
        # Has it changed?
        if self.comboGrid.get(row, col)[3] != bestMatch:
            changed = True
            # print(self.comboGrid.get(row, col), bestMatch)
            self.comboGrid.put(row, col, bestMatch)
            self.mockupImg[startY:endY, startX:endX] = self.compositeAdj(row, col)
        else:
            changed = False
            # self.comboGrid.clean(row, col)

        self.ditherImg = self.applyDither(row, col)
        return changed

        # Try compositing different characters onto a copy of the mockupSlice
    # Compare each to the targetSlice
    # Return the id of the best matching character
    def getBestAdj(self, row, col):

        # Score against temporary ditherImg created for this comparison
        def compare(row, col, ditherImg):
            startX, startY, endX, endY = self.getSliceBounds(row, col)
            targetSlice = ditherImg[startY:endY, startX:endX]
            mockupSlice = self.compositeAdj(row, col)
            # brighten targetSlice
            err = abs(np.average(targetSlice/self.maxGamma - mockupSlice))
            targetSlice = gammaCorrect(targetSlice, self.maxGamma)
            mse = np.sqrt(compare_mse(targetSlice, mockupSlice))
            # print('err:',err,'mse:',mse)
            return err + mse

        chars = self.charSet.getAll()
        scores = {}
        origGrid = self.comboGrid.grid.copy()
        for char in chars:
            self.stats['comparisonsMade'] += 1
            self.comboGrid.put(row, col, char.id)
            ditherImg = self.applyDither(row, col)
            # Score the composite
            scores[char.id] = compare(row, col, ditherImg)

        self.comboGrid.grid = origGrid

        # TODO return scores along with winning char ID for later use
        return min(scores, key=scores.get)


    def putBetter(self, row, col, k):
        self.stats['positionsVisited'] += 1
        startX, startY, endX, endY = self.getSliceBounds(row, col)
        
        bestMatch = self.getBestOfRandomK(row, col, k)

        if bestMatch:
            # print(self.comboGrid.get(row, col), bestMatch)
            changed = True
            self.comboGrid.put(row, col, bestMatch)
            self.mockupImg[startY:endY, startX:endX] = self.compositeAdj(row, col)
        else:
            # print("already good")
            changed = False
            # self.comboGrid.clean(row, col)

        self.ditherImg = self.applyDither(row, col)
        return changed


        # Only uses MSE
    def getBestOfRandomK(self, row, col, k=5, binned=False):
        
        # Score against temporary ditherImg created for this comparison
        def compare(row, col, ditherImg):
            startX, startY, endX, endY = self.getSliceBounds(row, col)
            targetSlice = ditherImg[startY:endY, startX:endX]
            mockupSlice = self.compositeAdj(row, col)
            # brighten targetSlice
            err = abs(np.average(targetSlice/self.maxGamma - mockupSlice))
            targetSlice = gammaCorrect(targetSlice, self.maxGamma)
            mse = np.sqrt(compare_mse(targetSlice, mockupSlice))
            # print('err:',err,'mse:',mse)
            return err + mse

        ditherImg = self.ditherImg
        curScore = compare(row, col, ditherImg)
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
            chars = np.random.choice(chars, k, replace=False)
        scores = {}
        origGrid = self.comboGrid.grid.copy()
        for char in chars:
            self.stats['comparisonsMade'] += 1
            self.comboGrid.put(row, col, char.id)
            ditherImg = self.applyDither(row, col)
            # Score the composite
            scores[char.id] = compare(row, col, ditherImg)

        self.comboGrid.grid = origGrid

        bestChoice = min(scores, key=scores.get)
        better = scores[bestChoice] < curScore
        return bestChoice if better else None


    # Return updated copy of the dither image based on current selection
    def applyDither(self, row, col, amount=0.5):
        # print("Begin dither")
        ditherImg = self.ditherImg.copy()

        startX, startY, endX, endY = self.getSliceBounds(row, col)
        ditherSlice = ditherImg[startY:endY, startX:endX]
        mockupSlice = self.mockupImg[startY:endY, startX:endX]
        ditherDone = np.zeros(ditherSlice.shape, dtype=np.bool)

        # Li dither by pixel
        residual = 0
        h, w = ditherSlice.shape
        K = 2.6 # Hyperparam
        M = 5   # Mask size
        c = M//2

        # Calculate error between chosen combo and target subslice
        # Per pixel
        for row in range(len(ditherSlice)):
            for col in range(len(ditherSlice[row])):
                actual = mockupSlice[row, col]
                target = ditherSlice[row, col]
                error = target - actual
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

                weightIdx = []
                for i, j, dist in adjIdx:
                    adjVal = ditherSlice[i, j]
                    # Darken slices which are already darker, and vice-versa
                    # Affect closer slices more
                    weight = (adjVal if error > 0 else 255 - adjVal) / (dist**K)
                    weightIdx.append((i, j, weight, adjVal))
                
                totalWeight = np.sum([weight for _, _, weight, _ in weightIdx])
                for i, j, weight, beforeVal in weightIdx:
                    # Normalize weights since not all slices will be adjustable
                    weight /= totalWeight
                    # Overall we want to reach this level with the slice:
                    desiredVal = beforeVal + error*weight + residual
                    # Apply corrections per pixel
                    correction = (desiredVal - beforeVal)
                    ditherSlice[i, j] = min(255, max(0, ditherSlice[i, j] + correction))
                    afterVal = ditherSlice[i, j]
                    residual = desiredVal - afterVal
                    # print(beforeVal, desiredVal - afterVal)

        # print("end dither")
        return ditherImg
        

    # Uses combos to store already composited "full" (all 4 layers)
    # If combo not already generated, add it to comboSet.
    # Returns mockupImg slice
    def compositeAdj(self, row, col):
        
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
        img = np.full((self.comboH*2, self.comboW*2), 255, dtype='uint8')
        img[:img.shape[0]//2, :img.shape[1]//2] = qs[0].img
        img[:img.shape[0]//2, img.shape[1]//2:] = qs[1].img
        img[img.shape[0]//2:, :img.shape[1]//2] = qs[2].img
        img[img.shape[0]//2:, img.shape[1]//2:] = qs[3].img
        return img


    def generateLayers(self, compareModes=['m'], numAdjustPasses=0,
                        show=True, mockupFn='mp_untitled', gamma=1,
                        randomInit=False, randomOrder=False):

        def dirtyLinearPositions(randomize=False):
            positions = []
            for layerID in [0, 3, 1, 2]:
                startRow = 0
                startCol = 0
                endRow = self.rows - 1
                endCol = self.cols - 1
                if layerID in [2, 3]:
                    startRow = 1
                if layerID in [1, 3]:
                    startCol = 1
                for row in range(startRow, endRow, 2):
                    for col in range(startCol, endCol, 2):
                        if self.comboGrid.isDirty(row,col):
                            positions.append((row, col))
                positions.append(None)
            if randomize:
                np.random.shuffle(positions)
            return positions

        def setupFig():
            fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
            return fig, ax

        modeDict = {
            'm':'mse',
            's':'ssim',
            'b':'blend'
        }
        compareModes = [modeDict[c] for c in compareModes]
        self.maxGamma = gamma
        self.compareMode = compareModes.pop(0)
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
                # self.comboGrid.printDirty()
                # print(self.comboGrid)
                self.positions += dirtyLinearPositions(randomize=randomOrder)
                # print("dirty:", len(self.positions))
            pos = self.positions.pop(0)
            if pos is None:
                self.ditherImg = self.targetImg.copy()
                return
            row, col = pos
            # if self.putBestAdj(row, col):
            if self.putBetter(row, col, 5): # best of k random
            # if self.putBetter(row, col, 1): # first random better
                ax[0].clear()
                ax[0].imshow(self.mockupImg, cmap='gray')
            ax[1].clear()
            ax[1].imshow(self.ditherImg, cmap='gray')

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