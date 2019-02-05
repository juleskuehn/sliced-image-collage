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
        self.img = img


class CharSet:
    def __init__(self, chars):
        self.chars = [Char(0,0,i,char) for i, char in enumerate(chars)]
    
    def get(self, i):
        return self.chars[i-1]