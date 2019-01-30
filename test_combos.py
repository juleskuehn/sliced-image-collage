import numpy as np

from combo import Combo, ComboSet
from combo_grid import ComboGrid

def main():
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
    s9 = ComboSet(numChars=9)
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


if __name__ == "__main__":
    main()