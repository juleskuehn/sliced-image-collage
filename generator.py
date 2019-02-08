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

    def getSliceBounds(self, row, col):
        startY = row * self.comboH
        startX = col * self.comboW
        endY = (row+2) * self.comboH
        endX = (col+2) * self.comboW
        return startX, startY, endX, endY


    # Finding best character that is BR at (row, col)
    def putBest(self, row, col):
        startX, startY, endX, endY = self.getSliceBounds(row, col)
        targetSlice = self.targetImg[startY:endY, startX:endX]
        mockupSlice = self.mockupImg[startY:endY, startX:endX]
        if not (self.compareMode == 'ssim' and self.overtype == 1):
            # Brighten the target depending on how many layers have been typed
            base = 0.25
            step = 0.25
            gamma = min((base*self.overtype)+step*self.numLayers, 1)
            # print(gamma)
            targetSlice = gammaCorrect(targetSlice, gamma)
        # Get ID of best match
        bestMatch = self.getBest(targetSlice, mockupSlice) 
        self.comboGrid.put(row, col, bestMatch)
        self.mockupImg[startY:endY, startX:endX] = self.composite(
                        mockupSlice, self.charSet.getByID(bestMatch).cropped)


    def composite(self, img1, img2):
        def toFloat(img):
            return np.array(img / 255, dtype="float32")

        return np.array(toFloat(img1) * toFloat(img2) * 255, dtype='uint8')

    # Try compositing different characters onto a copy of the mockupSlice
    # Compare each to the targetSlice
    # Return the id of the best matching character
    def getBest(self, targetSlice, mockupSlice):
        # TODO speed up search by taking advantage of sorted order
        chars = self.charSet.getAll()
        scores = {}
        scores2 = {}
        for char in chars:
            newMockup = self.composite(mockupSlice, char.cropped)
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

        if self.compareMode == 'blend':
            fMSE=self.numLayers/sum(scores.values())
            fSSIM=max(0,(4-self.numLayers))/sum(scores2.values())
            for k in scores:
                scores[k] = scores[k]*fMSE + scores2[k]*fSSIM

        # TODO return scores along with winning char ID for later use
        return min(scores, key=scores.get)
        # return min(chars, key=lambda x: (
        #     abs(x.avg*0.5 - np.average(targetSlice))
        #     )).id


    def generateLayers(self, compareModes=['mse']):
        
        self.compareMode = compareModes.pop(0)
        def linearPositions(layerID):
            startRow = 0
            startCol = 0
            endRow = self.rows - 1
            endCol = self.cols - 1
            if layerID in ['BL', 'BR']:
                startRow = 1
            if layerID in ['TR', 'BR']:
                startCol = 1
            positions = []
            for row in range(startRow, endRow, 2):
                for col in range(startCol, endCol, 2):
                    positions.append((row, col))
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

        # For top left layer, start at 0,0. For bottom left 1,0. Etc.
        # Using None to indicate when we are switching to another layer
        positions = linearPositions('TL') + [None]
        positions += linearPositions('BR') + [None]
        positions += linearPositions('TR') + [None]
        positions += linearPositions('BL') + [None]

        self.positions = positions[:]

        fig, ax1 = setupFig()

        def gen():
            while len(self.positions) > 0:
                yield 0

        def animate(frame):
            pos = self.positions.pop(0)
            if pos == None:
                # New staggered layer
                self.numLayers += 1
                print('Finished layer', self.numLayers)
                if len(self.positions) == 0 and len(compareModes) > 0:
                    # New overtype set (another 4 layers)
                    self.overtype += 1
                    print('Starting overtype', self.overtype)
                    self.numLayers = 0
                    self.compareMode = compareModes.pop(0)
                    self.positions = positions[:]
                if len(self.positions) > 0:
                    pos = self.positions.pop(0)
                else:
                    return
            row, col = pos
            self.putBest(row, col)
            ax1.clear()
            ax1.imshow(self.mockupImg, cmap='gray')

        numFrames = (len(positions)-4)*(len(compareModes)+1)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Jules Kuehn'), bitrate=1800)
        ani = animation.FuncAnimation(fig, animate, repeat=False, frames=numFrames, interval=1)
        # ani.save('mp_10_ssmm.mp4', writer=writer)
        plt.show()
        return self.comboGrid
        

    def generatePriorityOrder(self, preview=True):
        def genM():
            return genMockup(self.comboGrid, self.comboSet, self.targetShape, self.targetPadding)
        
        priorityPositions = self.calcPriorityPositions()

        # for row, col in priorityPositions:
        #     self.putBest(row, col)
        #     if preview:
        #         #create image plot
        #         im1 = ax1.imshow(genM(),cmap='gray')
        #         plt.show()


        def gen():
            while len(priorityPositions) > 0:
                yield 0

        def animate(frame):
            row, col = priorityPositions.pop()
            self.putBest(row, col)
            ax1.clear()
            ax1.imshow(genM(),cmap='gray')

        if preview:
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Jules Kuehn'), bitrate=1800)
            ani = animation.FuncAnimation(fig, animate, frames=gen, interval=1)
            # ani.save('animation.mp4', writer=writer)
            plt.show()
        else:
            for row, col in priorityPositions:
                self.putBest(row, col)
        
        return self.comboGrid