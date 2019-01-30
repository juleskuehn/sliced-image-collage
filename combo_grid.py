import numpy as np
from combo import Combo


class ComboGrid:
    # A grid of shape (rows, cols) of Combos.
    # A typed image with 20 rows and 20 cols would need a grid (21, 21)
    # 1 is the space (empty) character. It differs from having no constraint.
    def __init__(self, rows, cols):
        self.grid = np.array(
                            [[Combo(None, None, None, None)
                            for _ in range(cols)]
                            for _ in range(rows)], dtype=object)
        # Constrain the edges to the space character
        for combo in self.grid[0,:]:
            combo.TL = 1
            combo.TR = 1
        for combo in self.grid[-1,:]:
            combo.BL = 1
            combo.BR = 1
        for combo in self.grid[:,0]:
            combo.TL = 1
            combo.BL = 1
        for combo in self.grid[:,-1]:
            combo.TR = 1
            combo.BR = 1

    # row & col are coords for the combo slice under consideration
    def getConstraints(self, row, col):
        # Get constraints from surrounding combos
        TL = (self.grid[row-1, col-1].BR or
                self.grid[row-1, col].BL or
                self.grid[row, col-1].TR
                if (row > 0 and col > 0) else 1)
        TR = (self.grid[row-1, col+1].BL or
                self.grid[row-1, col].BR or
                self.grid[row, col+1].TL
                if row > 0 and col < self.grid.shape[1] - 1 else 1)
        BL = (self.grid[row+1, col-1].TR or
                self.grid[row+1, col].TL or
                self.grid[row, col-1].BR
                if row < (self.grid.shape[0] - 1) and col > 0 else 1)
        BR = (self.grid[row+1, col+1].TL or
                self.grid[row+1, col].TR or
                self.grid[row, col+1].BL
                if row < (self.grid.shape[0] - 1)
                and col < (self.grid.shape[1] - 1) else 1)
        return Combo(TL, TR, BL, BR)

    def put(self, row, col, combo):
        # Puts the combo into the map, if it fits constraints
        if combo.matchesConstraints(self.getConstraints(row, col)):
            self.grid[row, col] = combo
            return True
        return False

    def fillRandom(self, comboSet):
        print("Randomly filling", self.grid.shape, "grid with valid random combos")
        tries = 0
        combos = comboSet.byIdx
        randomPositions = [(i, j)
                            for i in range(self.grid.shape[0])
                            for j in range(self.grid.shape[1])]
        np.random.shuffle(randomPositions)
        for i, j in randomPositions:
            constraints = self.getConstraints(i, j)
            if constraints.isFull():
                self.grid[i,j] = constraints
                continue
            np.random.shuffle(combos)
            for combo in combos:
                tries += 1
                if self.put(i, j, combo):
                    break
        print("Done - took", tries, "tries.")

    def __str__(self):
        s = '   '
        for col in range(self.grid.shape[1]):
            s += '   ' + str(col) + '  '
        divider = '   ' + '-' * (self.grid.shape[1] * 6 + 1)
        s += '\n' + divider + '\n'
        for row in range(self.grid.shape[0]):
            s1 = ' ' + str(row) + ' | '
            for col in range(self.grid.shape[1]):
                s1 += f'{self.grid[row, col].TL} {self.grid[row, col].TR} | '
            s2 = '   | '
            for col in range(self.grid.shape[1]):
                s2 += f'{self.grid[row, col].BL} {self.grid[row, col].BR} | '
            s += s1 + '\n' + s2 + '\n' + divider + '\n'
        return s