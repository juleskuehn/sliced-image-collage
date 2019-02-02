import numpy as np
from skimage.measure import compare_ssim, compare_mse
from math import inf
from combo import Combo

class Selector:
    
    def __init__(self, comboSet):
        self.comboSet = comboSet
        self.comboH, self.comboW = comboSet.byIdx[0].img.shape

    


    def bestMSE(self, targetImgSlice, constraints):
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u.flatten(), v2_u.flatten()), -1.0, 1.0))

        # print('cons:',constraints)
        
        # Get slice of combos that are valid for these constraints
        validCombos = self.comboSet.byChars[
            constraints.TL or None:constraints.TL+1 if constraints.TL else None,
            constraints.TR or None:constraints.TR+1 if constraints.TR else None,
            constraints.BL or None:constraints.BL+1 if constraints.BL else None,
            constraints.BR or None:constraints.BR+1 if constraints.BR else None]
        # [print(combo) for combo in validCombos.flatten()]
        # If only one slice is valid, we're done.
        if type(validCombos) == Combo:
            bestCombo = validCombos
            # print("only one valid combo")
            # if bestCombo != constraints:
            #     print("error!!!!")
        # Otherwise, find the best
        else:
            div = targetImgSlice.shape[0]*targetImgSlice.shape[1]*64
            # print(len(validCombos.flatten()), "combos valid here")
            bestCombo = None
            bestErr = inf
            for combo in validCombos.flatten():
                if combo is None:
                    continue
                err = compare_mse(combo.img, targetImgSlice)/div
                # print(err)
                err += angle_between(combo.img, targetImgSlice)
                # print(err,'with angle')
                if err < bestErr:
                    bestCombo = combo
                    bestErr = err
        # print('best:',bestCombo)
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