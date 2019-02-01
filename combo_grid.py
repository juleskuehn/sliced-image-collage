import numpy as np
from combo import Combo


class ComboGrid:
    # A grid of shape (rows, cols) of Combos.
    # 1 is the space (empty) character. It differs from having no constraint.
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
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


    def put(self, row, col, combo):
        self.grid[row, col] = Combo(combo.TL, combo.TR, combo.BL, combo.BR)
        # print(self)
        if col > 0:
            self.grid[row, col - 1].TR = combo.TL
            self.grid[row, col - 1].BR = combo.BL
        if col < self.cols - 1:
            self.grid[row, col + 1].TL = combo.TR
            self.grid[row, col + 1].BL = combo.BR
        if row > 0:
            self.grid[row - 1, col].BL = combo.TL
            self.grid[row - 1, col].BR = combo.TR
            if col > 0:
                self.grid[row - 1, col - 1].BR = combo.TL
            if col < self.cols - 1:
                self.grid[row - 1, col + 1].BL = combo.TR
        if row < self.rows - 1:
            self.grid[row + 1, col].TL = combo.BL
            self.grid[row + 1, col].TR = combo.BR
            if col > 0:
                self.grid[row + 1, col - 1].TR = combo.BL
            if col < self.cols - 1:
                self.grid[row + 1, col + 1].TL = combo.BR


    def __str__(self):
        s = '   '
        for col in range(self.grid.shape[1]):
            s += '   ' + str(col) + ' '
        divider = '   ' + '-' * (self.grid.shape[1] * 5 + 1)
        s += '\n' + divider + '\n'
        for row in range(self.grid.shape[0]):
            s1 = ' ' + str(row) + ' | '
            for col in range(self.grid.shape[1]):
                s1 += f'{self.grid[row, col].TL or 0} {self.grid[row, col].TR or 0}  '
            s2 = '   | '
            for col in range(self.grid.shape[1]):
                s2 += f'{self.grid[row, col].BL or 0} {self.grid[row, col].BR or 0}  '
            s += s1 + '\n' + s2 + '\n\n'
        return s