import numpy as np
import matplotlib.pyplot as plt
import math
import random
from timeit import Timer
from annoy import AnnoyIndex
from matplotlib.image import imread, imsave
from PIL import Image
import cv2
from math import inf
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr

from prep_charset_simple import chop_charset

import sys
args = sys.argv

if len(args) < 7:
    print("Usage: python simple_selector.py sourceImg targetImg slicesX slicesY rowLength mode [distance mode] [blankSpace]")
    print("Mode can be: nn, ssim, mse, combo")
    print('Metric can be: "angular", "euclidean", "manhattan", "hamming", or "dot"')
    exit()

sourceImg = args[1]
targetImg = args[2]
slicesX = int(args[3])
slicesY = int(args[4])
rowLength = int(args[5])
mode = args[6]
if len(args) > 7 and mode == 'nn':
    dist = args[7]
else:
    dist = 'angular'
if len(args) > 7 and mode == 'combo':
    kBest = int(args[7])
else:
    kBest = 5

blankSpace = not 'noblank' in args

# Photo is a grid of characters, 10 characters wide
# photoFn = 'maisie-williams.png'
photoFn = targetImg
photo = cv2.imread(photoFn, cv2.IMREAD_GRAYSCALE)
# photo = imread(photoFn)[:,:,0]*255

# print(photo.shape)

# plt.imshow(photo, cmap='Greys_r')
# plt.show()

# charsX = 20
charsX = rowLength
# # Get character set and related info
# cropped, padded, (xPad,yPad), propChange = chop_charset(
#                 fn='hermes4.png', numX=80, numY=8, startX=0.62, startY=0.26,
#                 xPad=5, yPad=5, shrink=2)

# Get character set and related info
cropped, padded, (xPad,yPad), (xChange,yChange) = chop_charset(
                fn=sourceImg, numX=slicesX, numY=slicesY, startX=0, startY=0,
                xPad=0, yPad=0, shrink=1, blankSpace=blankSpace)

print("input photo has shape", photo.shape)
# Dimensions of a single char
charHeight, charWidth = cropped[0].shape
dim = charWidth * charHeight

def buildModel(trees=10):
  charset = AnnoyIndex(dim, dist)
  for i, char in enumerate(cropped):
    charset.add_item(i, np.ndarray.flatten(char))
  charset.build(trees)
  charset.save('charset.ann')


charset = AnnoyIndex(dim, dist)
# Either build the model or load it
buildModel(10)
charset.load('charset.ann')


# Input photo will be resized to match a multiple of character width exactly
# Height will be scaled to correct for propChange
# Height will be padded to match a multiple of character height exactly
def resizePhoto(im, rowLength):
    inHeight, inWidth = im.shape
    outWidth = rowLength * charWidth
    outHeight = round((outWidth / inWidth) * inHeight * (xChange/yChange))
    # outHeight = round(scale * inHeight) # No propChange for our experiment here
    im = cv2.resize(im, dsize=(outWidth, outHeight), interpolation=cv2.INTER_CUBIC)
    # Pad outHeight so that it aligns with a character boundary
    print("resized photo has shape", im.shape)
    if outHeight % charHeight != 0:
        newHeight = (outHeight//charHeight + 1) * charHeight
        blank = np.full((newHeight, outWidth), 255, dtype='uint8')
        blank[0:outHeight] = im
        im = blank
        print("adding padding to", im.shape)
    else:
        newHeight = inHeight
    return im, newHeight-outHeight


# Resize photo to 10 characters wide
rphoto, photoPadding = resizePhoto(photo, charsX)
# print(rphoto.shape)


# plt.imshow(rphoto, cmap='Greys_r')
# plt.show()

def getSimilar(v, mode):
    maxIdx = None
    maxScore = -inf
    score = 0
    if mode == 'combo':
        # Select best MSE from NN: angular top 10
        bestNN = charset.get_nns_by_vector(np.ndarray.flatten(v), kBest)
        for i in bestNN:
            score = -compare_mse(v, cropped[i])
            if score > maxScore:
                maxScore = score
                maxIdx = i
        return maxIdx
    for i, im in enumerate(cropped):
        if mode == 'ssim':
            score, _ = compare_ssim(v, im, full=True)
        elif mode == 'mse':
            score = -compare_mse(v, im)
        if score > maxScore:
            maxScore = score
            maxIdx = i
    return maxIdx
    
# Now with the appropriately sized photo,
# Generate typable by finding closest approximations for each char slice in photo
def genTypable(photo, mode):
    height, width = photo.shape
    typable = np.zeros((height//charHeight, width//charWidth),dtype=object)
    for y in range(typable.shape[0]):
        for x in range(typable.shape[1]):
            startY = y*charHeight
            startX = x*charWidth
            endY = (y+1)*charHeight
            endX = (x+1)*charWidth
            if endY < height and endX < width:
                v = photo[startY:endY, startX:endX]
            elif endY < height:
                v = photo[startY:endY, startX:]
            elif endX < width:
                v = photo[startY:, startX:endX]
            else:
                v = photo[startY:, startX:]
            # print(x,y)
            if mode == 'nn':
                chosenIdx = charset.get_nns_by_vector(np.ndarray.flatten(v), 1)[0]
            elif mode in ['ssim', 'mse', 'combo']:
                chosenIdx = getSimilar(v, mode)
            else:
                print("bad mode")
                exit()

            # print(chosenIdx)
            typable[y,x] = chosenIdx
    return typable


t = genTypable(rphoto, mode)
print(t)

# Returns a mockup image, with the same size as the original image (for evaluation)
def genMockup(typable):
    tHeight, tWidth = typable.shape
    # Generate output image
    mockup = np.zeros((tHeight*charHeight, tWidth*charWidth), dtype='uint8')

    for y in range(tHeight):
        for x in range(tWidth):
            startY = y*charHeight
            startX = x*charWidth
            endY = (y+1)*charHeight
            endX = (x+1)*charWidth
            if endY < tHeight*charHeight and endX < tWidth*charWidth:
                mockup[startY:endY, startX:endX] = cropped[typable[y,x]]
            elif endY < tHeight*charHeight:
                mockup[startY:endY, startX:] = cropped[typable[y,x]]
            elif endX < tWidth*charWidth:
                mockup[startY:, startX:endX] = cropped[typable[y,x]]
            else:
                mockup[startY:, startX:] = cropped[typable[y,x]]

    # Crop the mockup to match input image
    if photoPadding > 0:
        mockup = mockup[:-photoPadding, :]
        print("Cropped to", mockup.shape)

    # Rescale the mockup by 1/propChange
    # resized = cv2.resize(mockup, dsize=(round(tWidth*charWidth*xChange), round(tHeight*charHeight*yChange)), interpolation=cv2.INTER_CUBIC)
    # Resize to match input photo
    resized = cv2.resize(mockup, dsize=(photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_CUBIC)
    print("mockup has shape", resized.shape)

    return resized

m = genMockup(t)

# imsave(f'mockup/mockup_{mode}_{charsX}.png', m, cmap='Greys_r')
if mode != 'nn':
    dist = ''
else:
    dist = '_' + dist
if mode != 'combo':
    kBest = ''
else:
    kBest = '_best' + str(kBest)

print("writing file:")
mockupFn = f'mockup/mockup_{charsX}_{mode}{dist}{kBest}.png'
print(mockupFn)
cv2.imwrite(mockupFn, m)