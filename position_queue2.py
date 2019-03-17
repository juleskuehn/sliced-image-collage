from pqdict import pqdict
from random import random
from generator_utils import scoreAnn, scoreMse, compare

class PositionQueue:
    def __init__(self, generator):
        self.queue = pqdict({}, key=lambda x: x[0])
        self.generator = generator
        self.changed = {}
        # If k == 0, simply choose the best. Otherwise, increasing from k
        self.k = 0
        # TODO expose hyperparam
        self.maxFlips = 100
        print("Initializing priority queue")
        for row in range(generator.rows - 1):
            for col in range(generator.cols - 1):
                position = (row, col)
                self.add(position)
        # self.generator.comboGrid.initChanged()

    def add(self, position):
        bestScore, bestId = self.score(position)
        row, col = position
        flips = self.generator.comboGrid.flips[row, col]
        # print(row, col, flips)
        if flips > self.maxFlips:
            return
        self.queue[position] = (bestScore, (position, bestId))

    # Returns a tuple (position, bestId)
    def remove(self):
        # row, col = self.queue.topitem()[0]
        # print(self.queue.topitem()[1])
        # Update priorities of surrounding items
        return self.queue.popitem()[1][1]

    def update(self, row, col):
        if row > 0 and col > 0:
            self.add((row-1, col-1))
        if row > 0:
            self.add((row-1, col))
        if row > 0 and col+2 < self.generator.cols:
            self.add((row-1, col+1))
        if col > 0:
            self.add((row, col-1))
        if col+2 < self.generator.cols:
            self.add((row, col+1))
        if row+2 < self.generator.rows and col > 0:
            self.add((row+1, col-1))
        if row+2 < self.generator.rows:
            self.add((row+1, col))
        if row+2 < self.generator.rows and col+2 < self.generator.cols:
            self.add((row+1, col+1))

    def score(self, position):
        row, col = position
        oldScore = compare(self.generator, row, col)
        newScore, bestId = scoreMse(self.generator, row, col, k=self.k)
        if newScore - oldScore < 0:
            return newScore - oldScore, bestId
        else:
            return 0, self.generator.comboGrid.get(row, col)[3]