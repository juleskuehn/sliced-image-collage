from heapq import *
from generator_utils import scoreAnn, scoreMse, compare

class PositionQueue:
    def __init__(self, generator):
        self.queue = []
        self.generator = generator
        self.changed = {}
        print("Initializing priority queue")
        for row in range(generator.rows - 1):
            for col in range(generator.cols - 1):
                position = (row, col)
                self.add(position)
        # self.generator.comboGrid.initChanged()

    def add(self, position):
        bestScore, bestId = self.score(position)
        row, col = position
        # self.generator.comboGrid.setChanged(row, col)
        heappush(self.queue, (bestScore, (position, bestId)))

    # Returns a tuple (position, bestId)
    def remove(self):
        position = heappop(self.queue)
        row, col = position[1][0]
        # if self.generator.comboGrid.isChanged(row, col):
        score, bestId = self.score((row, col))
        # Is the score still the best?
        if score > self.queue[0][0]:
            heappush(self.queue, (score, ((row, col), bestId)))
            return self.remove()
        else:
        # self.add((row, col))
            print((score, ((row, col), bestId)))
            return ((row, col), bestId)

    def score(self, position):
        row, col = position
        oldScore = compare(self.generator, row, col)
        newScore, bestId = scoreMse(self.generator, row, col)
        return newScore - oldScore, bestId