import numpy as np


class Char:
    # Stores information about a single typed character
    # - top left corner position in original charset image
    # - chopped and scaled version of char image
    def __init__(self, x, y, idx, img):
        self.x = x
        self.y = y
        self.idx = idx
        self.h = len(img)
        self.w = len(img[0])
        self.TLq = img[:self.h//2, :self.w//2]
        self.TRq = img[:self.h//2, self.w//2:]
        self.BLq = img[self.h//2:, :self.w//2]
        self.BRq = img[self.h//2:, self.w//2:]

    def getImg(self):
        topRow = np.concatenate((self.TLq, self.TRq), axis=1)
        botRow = np.concatenate((self.BLq, self.BRq), axis=1)
        return np.concatenate((topRow, botRow), axis=0)


class Combo:
    # Stores indices of Char slices, index in ANN model
    # Optionally, stores composite image
    def __init__(self, TL, TR, BL, BR, idx=None, chars=None):
        self.TL = TL
        self.TR = TR
        self.BL = BL
        self.BR = BR
        self.idx = idx
        self.img = self.genComposite(chars) if chars else None

    def __str__(self):
        return str([[self.TL, self.TR], [self.BL, self.BR]])

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    # chars is the list of char slices from which combos were generated
    def genComposite(self, chars):
        def toFloat(char):
            return np.array(char / 255, dtype="float32")

        TLc = toFloat(chars[self.TL].BRq)
        TRc = toFloat(chars[self.TR].BLq)
        BLc = toFloat(chars[self.BL].TRq)
        BRc = toFloat(chars[self.BR].TLq)

        img = TLc * TRc * BLc * BRc

        return np.array(img * 255, dtype='uint8')

    def matchesConstraints(self, constraints):
        return ((self.TL == constraints.TL or constraints.TL == None)
            and (self.TR == constraints.TR or constraints.TR == None)
            and (self.BL == constraints.BL or constraints.BL == None)
            and (self.BR == constraints.BR or constraints.BR == None))

    def isFull(self):
        return np.all([self.TL, self.TR, self.BL, self.BR])


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

class ComboSet:
    # Container class with useful methods
    # Stores Combos in 4D Array for easy filtering by constraint
    # Also in a list by indices of the ANN model
    def __init__(self, numChars):
        self.byIdx = []
        self.byCombo = {}
        i = 0
        for a in range(1, numChars):
            for b in range(1, numChars):
                for c in range(1, numChars):
                    for d in range(1, numChars):
                        combo = Combo(a,b,c,d,idx=i)
                        self.byIdx.append(combo)
                        self.byCombo[combo] = i
                        i += 1
        print("Generated", i, "combos.")

    

testGrid = ComboGrid(3,3)

t = ComboGrid(3,3)

t.grid = np.empty((3,3), dtype=object)
A = 'A'
B = 'B'
C = 'C'
D = 'D'
t.grid[0,0] = Combo(0,0,0,A)
t.grid[0,1] = Combo(0,0,A,B)
t.grid[0,2] = Combo(0,0,B,0)
t.grid[1,0] = Combo(0,A,0,C)
t.grid[1,1] = Combo(A,B,C,D)
t.grid[1,2] = Combo(B,0,D,0)
t.grid[2,0] = Combo(0,C,0,0)
t.grid[2,1] = Combo(C,D,0,0)
t.grid[2,2] = Combo(D,0,0,0)

def checkGrid(t):
    for i in range(t.grid.shape[0]):
        for j in range(t.grid.shape[1]):
            match = t.grid[i,j].matchesConstraints(t.getConstraints(i,j))
            if not match:
                print(i, j, match)
                print('actual:', t.grid[i,j])
                print('constr:', t.getConstraints(i,j))


f = ComboGrid(3,3)

checkGrid(t)
print(t)
checkGrid(f)
print(f)
s9 = ComboSet(9)
g3 = ComboGrid(3, 3)
g3.fillRandom(s9)
checkGrid(g3)
print(g3)



g5 = ComboGrid(5, 5)
g5.fillRandom(s9)
checkGrid(g5)
print(g5)


g10 = ComboGrid(10, 10)
g10.fillRandom(s9)
checkGrid(g10)
print(g10)