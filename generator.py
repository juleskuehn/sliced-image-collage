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
        
        constraints = self.comboGrid.grid[row, col]
        bestMatch = self.selector.bestMSE(targetImgSlice, constraints)
        self.comboGrid.put(row, col, bestMatch)


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
            #     self.comboGrid.grid[row, col] = self.comboSet.byCombo[Combo(1,1,1,1)]
            #     continue
            self.putBest(row, col)
            
        return self.comboGrid