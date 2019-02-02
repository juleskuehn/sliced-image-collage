import numpy as np
import operator
import timeit
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from selector_mse_ssim import Selector
from combo import ComboSet, Combo
from combo_grid import ComboGrid
from char import Char
from kword_util import genMockup

class Generator:

    # Assumes targetImg has already been resized and padded to match combo dimensions
    def __init__(self, targetImg, comboSet, shapeliness=0.5, targetShape=None, targetPadding=None):
        self.targetImg = targetImg
        self.comboSet = comboSet
        self.comboH, self.comboW = comboSet.byIdx[0].img.shape
        self.rows = targetImg.shape[0] // self.comboH
        self.cols = targetImg.shape[1] // self.comboW
        self.targetShape = targetShape or targetImg.shape
        self.targetPadding = targetPadding or 0
        self.comboGrid = ComboGrid(self.rows, self.cols)
        self.selector = Selector(self.comboSet)
        self.shapeliness = shapeliness
        self.ditherResidual = 0
        self.times = {'dither': 0, 'putBest': 0, 'residual': 0}


    def getSliceBounds(self, row, col):
        startY = row * self.comboH  
        startX = col * self.comboW
        endY = (row+1) * self.comboH
        endX = (col+1) * self.comboW
        return startX, startY, endX, endY


    def putBest(self, row, col):

        def applyResidual():
            # Attempt to apply residual
            beforeVal = np.average(self.targetImg[startY:endY, startX:endX])
            desiredVal = beforeVal + self.ditherResidual
            self.targetImg[startY:endY, startX:endX] = cv2.add(
                self.targetImg[startY:endY, startX:endX], self.ditherResidual)
            afterVal = np.average(self.targetImg[startY:endY, startX:endX])
            self.ditherResidual = desiredVal - afterVal
            # print(self.ditherResidual)

        startX, startY, endX, endY = self.getSliceBounds(row, col)
        self.times['residual'] += timeit.timeit(applyResidual,number=1)

        def putBestMatch():
            # Find best match
            targetImgSlice = self.targetImg[startY:endY, startX:endX]
            constraints = self.comboGrid.get(row, col)
            bestMatch = self.selector.bestMSE(targetImgSlice, constraints)
            self.comboGrid.put(row, col, bestMatch)

        self.times['putBest'] += timeit.timeit(putBestMatch, number=1)
        
        def applyDither():
            self.applyDither(row, col)

        # self.times['dither'] += timeit.timeit(applyDither, number=1)


    def calcPriorityPositions(self):
        f = self.targetImg.shape[0]//7
        if f % 2 == 0:
            f += 1
        src = cv2.GaussianBlur(self.targetImg, (f, f), 0)
        # src = cv2.GaussianBlur(src, (f, f), 0)
        laplacian = cv2.Laplacian(src ,cv2.CV_64F)
        # cv2.imwrite('laplace.png', laplacian)

        d = {}
        for row in range(self.rows):
            for col in range(self.cols):
                startX, startY, endX, endY = self.getSliceBounds(row, col)
                targetSlice = laplacian[startY:endY, startX:endX]
                d[(row, col)] = np.sum(targetSlice)
        # print(d)
        return sorted(d, key=d.get, reverse=False)


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
        
        print(self.times)
        print(self.comboGrid)
        return self.comboGrid

    
    def testPriorityOrder(self):
        priorityPositions = self.calcPriorityPositions()
        pTestImg = np.zeros(self.targetImg.shape, dtype='uint8')
        i = 0
        for row, col in priorityPositions:
            startX, startY, endX, endY = self.getSliceBounds(row, col)
            pTestImg[startY:endY, startX:endX] = max(0, 255 - i*255/len(priorityPositions))
            i += 1
        return pTestImg


    # Applies dither *around* comboGrid[row, col] when that slice has already been chosen
    def applyDither(self, row, col):
        h, w = self.comboGrid.rows, self.comboGrid.cols
        K = 2.6 # Hyperparam
        M = 3   # Mask size
        c = M//2

        startX, startY, endX, endY = self.getSliceBounds(row, col)
        
        # Calculate error between chosen combo and target subslice
        actual = np.average(self.comboSet.byCombo[self.comboGrid.get(row, col)].img)
        target = np.average(self.targetImg[startY:endY, startX:endX])
        
        error = target - actual
        # print(error)
        
        # Get adjacent subslices which aren't chosen already, checking bounds
        adjIdx = [(i, j, np.linalg.norm(np.array([i,j])-np.array([row,col])))
                         for i in range(row-c, row+c+1)
                         for j in range(col-c, col+c+1)
                         if (j >= 0 and j < w and i >= 0 and i < h
                             and (i, j) != (row, col)
                             and not self.comboGrid.get(i, j).isDone())]

        weightIdx = []
        for i, j, dist in adjIdx:
            startX, startY, endX, endY = self.getSliceBounds(i, j)
            adjVal = np.average(self.targetImg[startY:endY, startX:endX])
            # Darken slices which are already darker, and vice-versa
            # Affect closer slices more
            weight = (adjVal if error > 0 else 255 - adjVal) / (dist**K)
            weightIdx.append((i, j, weight, adjVal))
        
        totalWeight = np.sum([weight for _, _, weight, _ in weightIdx])
        for i, j, weight, beforeVal in weightIdx:
            # Normalize weights since not all slices will be adjustable
            weight /= totalWeight
            # Overall we want to reach this level with the slice:
            desiredVal = beforeVal + error*weight + self.ditherResidual
            # Apply corrections per pixel
            correction = (desiredVal - beforeVal)
            startX, startY, endX, endY = self.getSliceBounds(i, j)
            self.targetImg[startY:endY, startX:endX] = cv2.add(self.targetImg[startY:endY, startX:endX], correction)
            afterVal = np.average(self.targetImg[startY:endY, startX:endX])
            self.ditherResidual = desiredVal - afterVal
            # print(beforeVal, desiredVal - afterVal)