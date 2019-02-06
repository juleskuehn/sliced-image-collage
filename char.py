import numpy as np
import cv2

class Char:
    # Stores information about a single typed character
    # Lots of redundant information, for speed of retrieval
    def __init__(self, paddedImg, cropSettings, id):
        self.id = id
        self.padded = paddedImg
        # Generate cropped image
        croppedSizeY = paddedImg.shape[0] - cropSettings['yPad']
        croppedSizeX = paddedImg.shape[1] - cropSettings['xPad']
        self.cropped = paddedImg[
            cropSettings['yCropPos']:cropSettings['yCropPos']+croppedSizeY,
            cropSettings['xCropPos']:cropSettings['xCropPos']+croppedSizeX
        ]
        self.shrunken = cv2.resize(
            self.cropped, dsize=(self.cropped.shape[1]//cropSettings['shrinkX'],
                            self.cropped.shape[0]//cropSettings['shrinkY']),
            interpolation=cv2.INTER_AREA
        )
        self.avg = np.average(self.cropped)
        h, w = self.cropped.shape
        self.TLavg = np.average(self.cropped[:h//2, :w//2])
        self.TRavg = np.average(self.cropped[:h//2, w//2:])
        self.BLavg = np.average(self.cropped[h//2:, :w//2])
        self.BRavg = np.average(self.cropped[h//2:, w//2:])

    def __str__(self):
        return (
            f'id: {self.id}, padded.shape: {self.padded.shape}, cropped.shape: {self.cropped.shape}, shrunken.shape: {self.shrunken.shape}\navg: {self.avg}, TLavg: {self.TLavg}, TRavg: {self.TRavg}, BLavg: {self.BLavg}, BRavg: {self.BRavg}\n'
        )

class CharSet:
    def __init__(self, paddedChars, cropSettings):
        self.chars = [
                        Char(charImg, cropSettings, i)
                        for i, charImg in enumerate(paddedChars)
                     ]
        self.sortedChars = sorted(self.chars, key=lambda x: x.avg, reverse=True)
    
    def get(self, i):
        return self.chars[i]

    def getById(self, id):
        return self.chars[i - 1]

    def getAll(self):
        return self.chars

    def getSorted(self):
        return self.sortedChars