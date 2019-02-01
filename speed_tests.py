import numpy as np
import timeit

from selector import Selector
from combo import ComboSet, Combo
from combo_grid import ComboGrid
from char import CharSet, Char
from generator import Generator

numChars = 10
charWidth = 10
charHeight = 20

targetWidth = 100
targetHeight = 200

# Simulate 20x10 pixel chars
chars = np.array(np.random.rand(numChars, charHeight, charWidth) * 255, dtype='uint8')

target = np.array(np.random.rand(targetHeight, targetWidth) * 255, dtype='uint8')

# comboSet = None
# def initComboSet():
#     comboSet = ComboSet(len(chars), charset=CharSet(chars))
# print('initComboSet x10):', timeit.timeit( initComboSet, number=10 ))
# 2.68 seconds with 10 chars (10k combos)
# 40 seconds with 20 chars (160k combos)

# selector = None
# def initSelector():
#     selector = Selector(comboSet)
# print('initSelector x10):', timeit.timeit( initSelector, number=1 ))
# 5.85 seconds with 10 chars (10k combos)
# 102 seconds with 20 chars (160k combos)
# Roughly linear

# generator = Generator(target, comboSet, shapeliness=0.5)

comboSet = ComboSet(len(chars), charset=CharSet(chars))
combo = comboSet.byIdx[np.random.randint(len(comboSet.byIdx))]
constraints = comboSet.byIdx[np.random.randint(len(comboSet.byIdx))]

comboGrid = ComboGrid(80,80)
def getConstraints():
    comboGrid.getConstraints(40,40)

print('getConstraints x10m):', timeit.timeit( getConstraints, number=10000000 ))
# 36 seconds for 10m iterations: not negligible
# 25 seconds for 10m iterations with speedups
# - no bounds checking   - no combo generation

# def checkMatchesConstraints():
    # return combo.matchesConstraints(constraints)

# print('checkMatchesConstraints x100m):', timeit.timeit( checkMatchesConstraints, number=100000000 ))
# 36 seconds with 10 chars (10k combos) for 100m iterations: basically instant
# selector = Selector(comboSet)

# randSlice = np.array(np.random.rand(charHeight//2, charWidth//2) * 255, dtype='uint8')
# def getSimilar():
#     selector.getSimilar(randSlice, 0.5)

# print('getSimilar x1000):', timeit.timeit( getSimilar, number=1000 ))
# 25 seconds with 10 chars (10k combos)
# 17 seconds without normalizing and blending metrics
# 8.5 seconds with one metric only (makes sense)
#   - note that 1000 calls to getSimilar is realistic
# 180 seconds with one metric only, as above, but 20 chars (160k combos)
#  - slightly worse than linear increase

# generator = Generator(target, comboSet)

# def runGenerator():
