import numpy as np
import operator

from selector_mse_ssim import Selector
from combo import ComboSet, Combo
from combo_grid import ComboGrid
from char import Char
import cv2

class Generator:

    # Assumes targetImg has already been resized and padded to match combo dimensions
    def __init__(self, targetImg, comboSet, shapeliness=0.5):
        self.targetImg = targetImg
        self.comboSet = comboSet
        self.comboH, self.comboW = comboSet.byIdx[0].img.shape
        self.rows = targetImg.shape[0] // self.comboH
        self.cols = targetImg.shape[1] // self.comboW
        self.comboGrid = ComboGrid(self.rows, self.cols)
        self.selector = Selector(self.comboSet)
        self.shapeliness = shapeliness


    def putBest(self, row, col):
        startY = row * self.comboH
        startX = col * self.comboW
        endY = (row+1) * self.comboH
        endX = (col+1) * self.comboW
        targetImgSlice = self.targetImg[startY:endY, startX:endX]
        
        constraints = self.comboGrid.get(row, col)
        bestMatch = self.selector.bestMSE(targetImgSlice, constraints)
        self.comboGrid.put(row, col, bestMatch)
        self.applyDither(row, col, targetImgSlice)


    def generatePriorityOrder(self):

        def calcPriorityPositions():
            laplacian = cv2.Laplacian(self.targetImg,cv2.CV_64F)
            cv2.imwrite('laplace.png', laplacian)

            d = {}
            for row in range(self.rows):
                for col in range(self.cols):
                    startY = row * self.comboH  
                    startX = col * self.comboW
                    endY = (row+1) * self.comboH
                    endX = (col+1) * self.comboW
                    targetSlice = laplacian[startY:endY, startX:endX]
                    d[(row, col)] = np.sum(targetSlice)
            return sorted(d, key=d.get, reverse=False)

        priorityPositions = calcPriorityPositions()
        i = 0
        for row, col in priorityPositions:
            # print(self.comboGrid)
            # i += 1
            # if i > 200:
            #     self.comboGrid.get(row, col) = self.comboSet.byCombo[Combo(1,1,1,1)]
            #     continue
            self.putBest(row, col)
            
        return self.comboGrid

    def applyDither(self, row, col, targetImgSlice):
        residual = 0
        h, w = self.comboGrid.rows, self.comboGrid.cols
        K = 2.6 # Hyperparam
        M = 3   # Mask size
        c = M//2
        numPixels = targetImgSlice.shape[0] * targetImgSlice.shape[1]

        # Calculate error between chosen combo and target subslice
        actual = np.sum(self.comboSet.byCombo[self.comboGrid.get(row, col)].img)
        target = np.sum(targetImgSlice) + residual
        error = (target - actual) / numPixels
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
            startY = i * self.comboH  
            startX = j * self.comboW
            endY = (i+1) * self.comboH
            endX = (j+1) * self.comboW
            adjVal = np.sum(self.targetImg[startY:endY, startX:endX]) / numPixels
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
            startY = i * self.comboH  
            startX = j * self.comboW
            endY = (i+1) * self.comboH
            endX = (j+1) * self.comboW
            self.targetImg[startY:endY, startX:endX] = cv2.add(self.targetImg[startY:endY, startX:endX], correction)
            afterVal = np.sum(self.targetImg[startY:endY, startX:endX]) / numPixels
            # residual = desiredVal - afterVal
            # print(beforeVal, desiredVal - afterVal)