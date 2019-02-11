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
        self.changed = []

    def getSliceBounds(self, row, col):
        startY = row * self.comboH
        startX = col * self.comboW
        endY = (row+2) * self.comboH
        endX = (col+2) * self.comboW
        return startX, startY, endX, endY


    # Finding best character that is BR at (row, col)
    def putBestAdj(self, row, col):
        startX, startY, endX, endY = self.getSliceBounds(row, col)
        targetSlice = self.targetImg[startY:endY, startX:endX]
        # Brighten the target to match maximum typable in first overtype
        if self.compareMode == 'mse':
            targetSlice = gammaCorrect(targetSlice, self.maxGamma[0])
        # Get ID of best match
        bestMatch = self.getBestAdj(targetSlice, row, col)
        # print(bestMatch)
        # Has it changed?
        
        if self.comboGrid.get(row, col)[3] != bestMatch:
            # print(self.comboGrid.get(row, col), bestMatch)
            self.comboGrid.put(row, col, bestMatch)
            self.mockupImg[startY:endY, startX:endX] = self.compositeAdj(row, col)
            return True
        else:
            self.comboGrid.clean(row, col)
            return False


    # Uses combos to store already composited "full" (all 4 layers)
    # If combo not already generated, add it to comboSet.
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


    # Try compositing different characters onto a copy of the mockupSlice
    # Compare each to the targetSlice
    # Return the id of the best matching character
    def getBestAdj(self, targetSlice, row, col):
        # TODO speed up search by taking advantage of sorted order
        chars = self.charSet.getAll()
        scores = {}
        scores2 = {}
        origGrid = self.comboGrid.grid.copy()
        for char in chars:
            self.comboGrid.put(row, col, char.id)
            newMockup = self.compositeAdj(row, col)
            # Score the composite
            if self.compareMode == 'mse':
                scores[char.id] = compare_mse(targetSlice, newMockup)
            elif self.compareMode == 'ssim':
                scores[char.id] = -1 * compare_ssim(targetSlice, newMockup)
            elif self.compareMode == 'blend':
                scores[char.id] = compare_mse(targetSlice, newMockup) 
                scores2[char.id] = -1 * compare_ssim(targetSlice, newMockup) + 1
            else:
                print('generator: invalid compareMode')
                exit()

        self.comboGrid.grid = origGrid

        if self.compareMode == 'blend':
            fMSE=self.numLayers/sum(scores.values())
            fSSIM=max(0,(4-self.numLayers))/sum(scores2.values())
            for k in scores:
                scores[k] = scores[k]*fMSE + scores2[k]*fSSIM

        # TODO return scores along with winning char ID for later use
        return min(scores, key=scores.get)


    def generateLayers(self, compareModes=['m'], numAdjustPasses=0,
                        show=True, mockupFn='mp_untitled'):
        
        modeDict = {
            'm':'mse',
            's':'ssim',
            'b':'blend'
        }
        compareModes = [modeDict[c] for c in compareModes]
        

        def dirtyLinearPositions(randomize=True):
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
            if randomize:
                np.random.shuffle(positions)
            return positions

        def setupFig():
            fig = plt.figure(frameon=False)
            fig.set_size_inches(5,5)
            ax1 = plt.Axes(fig, [0., 0., 1., 1.])
            ax1.set_axis_off()
            fig.add_axes(ax1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.axis('off')
            fig.subplots_adjust(bottom = 0)
            fig.subplots_adjust(top = 1)
            fig.subplots_adjust(right = 1)
            fig.subplots_adjust(left = 0)
            fig.gca().set_frame_on(False)
            return fig, ax1

        self.compareMode = compareModes.pop(0)
        # For top left layer, start at 0,0. For bottom left 1,0. Etc.
        # Using None to indicate when we are switching to another layer
        self.positions = dirtyLinearPositions()

        fig, ax1 = setupFig()
        self.adjustPass = 0

        def genFrames():
            while len(self.positions) > 0 or len(dirtyLinearPositions()) > 0:
                yield 1


        def animate(frame):
            if len(self.positions) == 0:
                print("Finished pass")
                self.comboGrid.printDirty()
                print(self.comboGrid)
                self.positions += dirtyLinearPositions()
                print("dirty:", len(self.positions))
            row, col = self.positions.pop(0)
            if self.putBestAdj(row, col):
                ax1.clear()
                ax1.imshow(self.mockupImg, cmap='gray')

        # numFrames = (len(self.positions)-4)*(len(compareModes)+1+numAdjustPasses)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Jules Kuehn'), bitrate=1800)
        ani = animation.FuncAnimation(fig, animate, repeat=False, frames=genFrames(), interval=20)
        if show:
            plt.show()
        else:
            ani.save(mockupFn+'.mp4', writer=writer)
        return self.comboGrid