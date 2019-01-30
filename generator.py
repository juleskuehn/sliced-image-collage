import numpy as np
import operator

from selector import Selector
from combo import ComboSet
from combo_grid import ComboGrid
from char import Char

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


    # Selects source slice (index) with lowest MSE vs target slice
    def getBest(self, row, col):

        constraints = self.comboGrid.getConstraints(row, col)

        def fitsConstraints(comboIdx):
            combo = self.comboSet.byIdx[comboIdx]
            return combo.matchesConstraints(constraints)

        startY = row * self.comboH
        startX = col * self.comboW
        endY = (row+1) * self.comboH
        endX = (col+1) * self.comboW

        targetImgSlice = self.targetImg[startY:endY, startX:endX]
        scoredCombos = self.selector.getSimilar(targetImgSlice, self.shapeliness)

        bestScore = 1
        bestComboIdx = None

        for key in scoredCombos:
            if fitsConstraints(key) and scoredCombos[key] < bestScore:
                bestScore = scoredCombos[key]
                bestComboIdx = key
        
        return bestComboIdx

    def putBest(self, row, col):
        self.comboGrid.put(row, col, self.comboSet.byIdx[self.getBest(row, col)])
        # self.comboGrid.grid[row, col] = self.comboSet.byIdx[self.getBest(row, col)]


    def generateLinearOrder(self):
        for row in range(self.rows):
            for col in range(self.cols):
                constraints = self.comboGrid.getConstraints(row, col)
                if constraints.isFull():
                    self.comboGrid.grid[row, col] = self.comboSet.byCombo[constraints]
                    continue
                self.putBest(row, col)
        return self.comboGrid

    def generateRandomOrder(self):
        randomPositions = [(row, col)
                            for row in range(self.rows)
                            for col in range(self.cols)]
        np.random.shuffle(randomPositions)
        for row, col in randomPositions:
            constraints = self.comboGrid.getConstraints(row, col)
            if constraints.isFull():
                self.comboGrid.grid[row, col] = self.comboSet.byCombo[constraints]
                continue
            self.putBest(row, col)
        return self.comboGrid