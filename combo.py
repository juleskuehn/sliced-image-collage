import numpy as np


class Combo:
    # Stores indices of Char slices and index of combo in ANN model
    # Optionally, stores composite image
    # Can also be used as a constraint (with no image or index)
    def __init__(self, TL, TR, BL, BR, idx=None, charset=None):
        self.TL = TL
        self.TR = TR
        self.BL = BL
        self.BR = BR
        self.idx = idx
        self.img = self.genComposite(charset) if charset else None

    def __str__(self):
        return str([[self.TL, self.TR], [self.BL, self.BR]])

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    # charset is the list of char slices from which combos were generated
    def genComposite(self, charset):
        def toFloat(char):
            return np.array(char / 255, dtype="float32")

        TLc = toFloat(charset.get(self.TL).BRq)
        TRc = toFloat(charset.get(self.TR).BLq)
        BLc = toFloat(charset.get(self.BL).TRq)
        BRc = toFloat(charset.get(self.BR).TLq)

        img = TLc * TRc * BLc * BRc

        return np.array(img * 255, dtype='uint8')

    def isDone(self):
        return np.all([self.TL, self.TR, self.BL, self.BR])


class ComboSet:
    # Container class with useful methods
    # Stores Combos in 4D Array for easy filtering by constraint
    # Also in a list by indices of the ANN model
    def __init__(self, charSet):
        self.byIdx = []
        self.byCombo = {}
        self.numChars = len(charSet.chars)
        n = self.numChars
        self.byChars = np.empty((n+1,n+1,n+1,n+1), dtype=object)
        self.size = n**4
        i = 0
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                for c in range(1, n + 1):
                    for d in range(1, n + 1):
                        combo = Combo(a,b,c,d,idx=i,charset=charSet)
                        self.byIdx.append(combo)
                        self.byCombo[combo] = combo
                        self.byChars[a,b,c,d] = combo
                        i += 1

        # [print(combo) for combo in self.byChars[1,1,1,:]]
        print("Generated", i, "combos.")