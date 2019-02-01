import numpy as np
from skimage.measure import compare_ssim, compare_mse
from math import inf
from combo import Combo

class Selector:
    
    def __init__(self, comboSet):
        self.comboSet = comboSet
        self.comboH, self.comboW = comboSet.byIdx[0].img.shape


    def bestMSE(self, targetImgSlice, constraints):
        
        TL = constraints.TL
        TR = constraints.TR
        BL = constraints.BL
        BR = constraints.BR
        
        # Get slice of combos that are valid for these constraints
        validCombos = self.comboSet.byChars[TL, TR, BL, BR]

        # If only one slice is valid, we're done.
        if type(validCombos) == Combo:
            bestCombo = validCombos
            # print("only one valid combo")
            # if bestCombo != constraints:
            #     print("error!!!!")
        # Otherwise, find the best
        else:
            # print(len(validCombos.flatten()), "combos valid here")
            bestCombo = None
            bestScore = inf
            for combo in validCombos.flatten():
                if combo is None:
                    continue
                score = compare_mse(combo.img, targetImgSlice)
                # score = 10
                if score < bestScore:
                    bestCombo = combo
                    bestScore = score
        # print(constraints)
        # print(bestCombo)
        # print('')
        return bestCombo


    def bestSSIM(self, targetImgSlice, c):
        # Get slice of combos that are valid for these constraints
        validCombos = self.comboSet.byChars[c.TL, c.TR, c.BL, c.BR]
        # If only one slice is valid, we're done.
        if type(validCombos) == Combo:
            return validCombos
        bestCombo = None
        bestScore = inf
        for combo in validCombos.flatten():
            if combo is None:
                continue
            score = compare_ssim(combo.img, targetImgSlice)
            if score < bestScore:
                bestCombo = combo
                bestScore = score
        return bestCombo


    def bestCombo(self, targetImgSlice, shapeliness=0.5):
        pass