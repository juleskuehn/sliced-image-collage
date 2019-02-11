import numpy as np
from combo import Combo


class ComboGrid:
    # A grid of shape (rows, cols) of Combos.
    # 1 is the space (empty) character. It differs from having no constraint.
    def __init__(self, rows, cols, random=0):
        self.rows = rows
        self.cols = cols
        # Intiialize grid to space character
        self.grid = np.array([[[1,1,1,1]
                                for _ in range(cols)]
                                for _ in range(rows)], dtype=object)
                                # Order of self.dirty is TL, TR, BL, BR
        self.dirty = np.array([[[1,1,1,1]
                                for _ in range(cols)]
                                for _ in range(rows)], dtype=np.bool)
        # Space character on edges never changes, so clear dirty bit
        self.dirty[0,:,:2] = 0
        self.dirty[-1,:,2:] = 0
        self.dirty[:,0,0] = 0
        self.dirty[:,0,2] = 0
        self.dirty[:,-1,1] = 0
        self.dirty[:,-1,3] = 0


    # Put charID at the bottom right of (row,col), bottom left of col+1, etc
    def put(self, row, col, charID):
        # if not self.isDirty(row, col):
        #     print("error! putting char at clean pos")
        #     exit()
        self.grid[row, col, 3] = charID
        self.grid[row+1, col, 1] = charID
        self.grid[row, col+1, 2] = charID
        self.grid[row+1, col+1, 0] = charID
        self.setDirty(row, col)


    def setDirty(self, row, col, isDirty=True):
        # Set dirty bits
        self.dirty[row, col, 3] = isDirty
        self.dirty[row+1, col, 1] = isDirty
        self.dirty[row, col+1, 2] = isDirty
        self.dirty[row+1, col+1, 0] = isDirty

    
    def isDirty(self, row, col):
        dirty = np.sum([1 for bit in self.dirty[row:row+2, col:col+2].flatten() if bit]) > 0
        # print(row, col, 'is', dirty)
        return dirty


    def clean(self, row, col):
        # print("Cleaning position", row, col)
        self.setDirty(row, col, False)


    def get(self, row, col):
        return self.grid[row, col]

    

    def __str__(self):
        s = '   '
        for col in range(self.grid.shape[1]):
            s += f'    {col:2} '
        divider = ' ' + '-' * (len(s))
        s += '\n' + divider + '\n'
        for row in range(self.grid.shape[0]):
            s1 = f' {row:2} | '
            for col in range(self.grid.shape[1]):
                s1 += f"{self.grid[row, col][0] or 0:2} {self.grid[row, col][1] or 0:2}  "
            s2 = '    | '
            for col in range(self.grid.shape[1]):
                s2 += f"{self.grid[row, col][2] or 0:2} {self.grid[row, col][3] or 0:2}  "
            s += s1 + '\n' + s2 + '\n\n'
        return s


    def printDirty(self):
        s = '   '
        for col in range(self.dirty.shape[1]):
            s += f'    {col:2} '
        divider = ' ' + '-' * (len(s))
        s += '\n' + divider + '\n'
        for row in range(self.dirty.shape[0]):
            s1 = f' {row:2} | '
            for col in range(self.dirty.shape[1]):
                s1 += f"{self.dirty[row, col, 0] or 0:2} {self.dirty[row, col, 1] or 0:2}  "
            s2 = '    | '
            for col in range(self.dirty.shape[1]):
                s2 += f"{self.dirty[row, col, 2] or 0:2} {self.dirty[row, col, 3] or 0:2}  "
            s += s1 + '\n' + s2 + '\n\n'
        print(s)