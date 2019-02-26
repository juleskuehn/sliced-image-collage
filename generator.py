import numpy as np
import operator
import timeit
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.measure import compare_ssim, compare_mse
import pickle

from combo import ComboSet, Combo
from combo_grid import ComboGrid
from char import Char
from kword_utils import genMockup, gammaCorrect
from generator_utils import putBetter, initRandomPositions, putSimAnneal, evaluateMockup

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

class Generator:

    # Assumes targetImg has already been resized and padded to match combo dimensions
    def __init__(self, targetImg, shrunkenTargetImg, charSet, shapeliness=0.5,
                        targetShape=None, targetPadding=None, shrunkenTargetPadding=None, dither=True):
        self.targetImg = targetImg
        print('t', targetImg.shape)
        self.shrunkenTargetImg = shrunkenTargetImg
        print('s', shrunkenTargetImg.shape)
        self.ditherImg = shrunkenTargetImg.copy()
        self.charSet = charSet
        self.comboSet = ComboSet()
        self.comboH, self.comboW = charSet.get(0).cropped.shape[0]//2, charSet.get(0).cropped.shape[1]//2
        self.shrunkenComboH, self.shrunkenComboW = charSet.get(0).shrunken.shape[0]//2, charSet.get(0).shrunken.shape[1]//2
        self.mockupRows = targetImg.shape[0] // self.comboH
        self.mockupCols = targetImg.shape[1] // self.comboW
        print('mockupRows', self.mockupRows, 'mockupCols', self.mockupCols)
        self.rows = shrunkenTargetImg.shape[0] // self.shrunkenComboH
        self.cols = shrunkenTargetImg.shape[1] // self.shrunkenComboW
        print('rows      ', self.rows,       'cols      ', self.cols)
        self.targetShape = targetShape or targetImg.shape
        self.mockupImg = np.full(targetImg.shape, 255, dtype='uint8')
        self.fixedMockupImg = np.full(targetImg.shape, 255, dtype='uint8')
        self.shrunkenMockupImg = np.full(shrunkenTargetImg.shape, 255, dtype='uint8')
        self.targetPadding = targetPadding or 0
        self.shrunkenTargetPadding = shrunkenTargetPadding or 0
        self.comboGrid = ComboGrid(self.rows, self.cols)
        self.compareMode = 'mse'
        self.numLayers = 0 # How many times has the image been typed
        self.overtype = 1 # How many times have all 4 layers been typed
        self.passNumber = 0
        self.gamma = 1
        self.stats = {
            'positionsVisited': 0,
            'comparisonsMade': 0
        }
        self.dither = dither
        self.boostK = 0
        # starting temperature for simulated annealing
        # not sure if this is a good starting point yet...
        # We will subtract comparisonsMade to a minimum of 0
        self.initTemp = 0.005
        self.minTemp = 0.00000001
        self.tempHistory = []
        self.psnrHistory = []


    def getTemp(self):
        return max(self.minTemp, self.initTemp - self.stats['positionsVisited']/(self.rows*self.cols*1000))


    def load_state(self, fn):
        state = load_object(fn)
        self.comboGrid = ComboGrid(self.rows, self.cols)
        # self.comboGrid.initDirty()
        self.fixedMockupImg = state['mockupImg']
        self.mockupImg = state['mockupImg']
        self.passNumber = state['passNumber']
        print("Resuming at pass", self.passNumber + 1)


    def generateLayers(self, compareMode='mse', numAdjustPasses=0,
                        show=True, mockupFn='mp_untitled', gamma=1,
                        randomInit=False, randomOrder=False):

        def dirtyLinearPositions(randomOrder=False, zigzag=True):
            positions = []
            for layerID in [0, 3, 1, 2]:
                startIdx = len(positions)
                # r2l = False if np.random.rand() < 0.5 else True
                startRow = 0
                startCol = 0
                endRow = self.rows - 1
                endCol = self.cols - 1
                if layerID in [2, 3]:
                    startRow = 1
                    # r2l = True
                if layerID in [1, 3]:
                    startCol = 1
                for row in range(startRow, endRow, 2):
                    for col in range(startCol, endCol, 2):
                        if self.dither and self.comboGrid.isDitherDirty(row, col):
                            positions.append((row, col))
                        elif self.comboGrid.isDirty(row,col):
                        # if self.comboGrid.isDirty(row,col):
                            positions.append((row, col))
                        else:
                            self.comboGrid.clean(row, col)
                # if r2l and zigzag:
                    # positions[startIdx:len(positions)] = positions[len(positions)-1:startIdx-1:-1]
                if len(positions) > 0:
                    positions.append(None)
            if randomOrder:
                np.random.shuffle(positions)
            # print(positions)
            # exit()
            return positions

        def setupFig():
            fig, ax = plt.subplots(1, 1)
            return fig, ax

        # self.maxGamma = gamma
        self.compareMode = compareMode
        self.gamma = gamma
        if self.compareMode == 'dither':
            print("!NO!U!N!OU!N!")
            self.dither = True
        else:
            self.dither = False
        self.positions = dirtyLinearPositions(randomOrder=randomOrder)
        # self.positions = []

        if randomInit:
            initRandomPositions(self)

        fig, ax = setupFig()
        # self.adjustPass = 0
        printEvery = 50

        def animate(frame):
            if frame % printEvery == 0:
                print(self.stats['positionsVisited'], 'positions visited')
                print(self.stats['comparisonsMade'], 'comparisons made')
                print(len(dirtyLinearPositions()), 'dirty positions remaining')
                print('Temperature: ', self.getTemp())
                self.psnrHistory.append(evaluateMockup(self))
                self.tempHistory.append(self.getTemp())
                print('---')
            if len(self.positions) == 0:
                print("Finished pass")
                # self.boostK += 2
                # self.comboGrid.printDirty()
                # print(self.comboGrid)
                # Clear at every new pass of 4:
                self.ditherImg = self.shrunkenTargetImg.copy()
                self.positions += dirtyLinearPositions(randomOrder==randomOrder)
                # print("dirty:", len(self.positions))
                if len(self.positions) == 0:
                    self.passNumber += 1
                    # There were no more dirty positions: finished!
                    print("Finished adjusting pass", self.passNumber, ", saving.")
                    save_object({
                        'mockupImg': self.mockupImg,
                        'comboGrid': self.comboGrid,
                        'passNumber': self.passNumber
                        }, 'pass_'+str(self.passNumber))
                    # self.comboGrid = ComboGrid(self.rows, self.cols)
                    # self.fixedMockupImg = self.mockupImg.copy()
                    # self.dither = self.compareMode == 'dither'
                    # self.positions = dirtyLinearPositions(randomize=randomOrder)
                    # # Need to reset combos because running out of memory?
                    # self.comboSet = ComboSet()

            pos = self.positions.pop(0)
            if pos is None:
                # Clear at every new overlap layer:
                # self.ditherImg = self.shrunkenTargetImg.copy()
                return
            row, col = pos
            # if self.putBestAdj(row, col):
            if putBetter(self, row, col, 40) or frame==0:
            # if putSimAnneal(self, row, col) or frame==0:
            # if self.putBetter(row, col, 1): # first random better
                ax.clear()
                ax.imshow(self.mockupImg, cmap='gray')
            # ax[1].clear()
            # ax[1].imshow(self.ditherImg if self.dither else self.targetImg, cmap='gray')

        # numFrames = (len(self.positions)-4)*(len(compareModes)+1+numAdjustPasses)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Jules Kuehn'), bitrate=1800)
        ani = animation.FuncAnimation(fig, animate, repeat=False, frames=1000000000, interval=1)
        if show:
            plt.show()
        else:
            ani.save(mockupFn+'.mp4', writer=writer)

        print("Finished!")
        print(self.stats['positionsVisited'], 'positions visited')
        print(self.stats['comparisonsMade'], 'comparisons made')
        
        # Prevent saving after initial error
        if self.stats['positionsVisited'] > 50:
            save_object({
                'mockupImg': self.mockupImg,
                'comboGrid': self.comboGrid,
                'passNumber': self.passNumber,
                'psnrHistory': self.psnrHistory,
                'tempHistory': self.tempHistory
                }, 'pass_'+str(self.passNumber)+'_end')

        return self.comboGrid