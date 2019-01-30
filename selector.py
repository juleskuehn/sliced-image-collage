import numpy as np
import operator
from annoy import AnnoyIndex

class Selector:
    
    def __init__(self, comboSet, trees=10):
        self.comboSet = comboSet
        self.comboH, self.comboW = comboSet.byIdx[0].img.shape

        print("Building angular model with", comboSet.size, "sub-slices...")
        self.angularNN = self.buildModel('angular', trees)

        print("Building euclidean model...")
        self.euclideanNN = self.buildModel('euclidean', trees)

        print("Done. Selector is ready.")


    def buildModel(self, distMetric, trees=10):
        dim = self.comboH * self.comboW
        model = AnnoyIndex(dim, metric=distMetric)
        for i, combo in enumerate(self.comboSet.byIdx):
            model.add_item(i, np.ndarray.flatten(combo.img))
        model.build(trees)
        model.save(distMetric+'.ann')
        return model


    def getSimilar(self, targetImgSlice, shapeliness):

        v = np.ndarray.flatten(targetImgSlice)
        
        aIndices, aScores = self.angularNN.get_nns_by_vector(v, self.comboSet.size, include_distances=True)
        eIndices, eScores = self.euclideanNN.get_nns_by_vector(v, self.comboSet.size, include_distances=True)

        maxAngular = np.max(aScores)
        minAngular = np.min(aScores)
        maxEuclidean = np.max(eScores)
        minEuclidean = np.min(eScores)
        
        aScores = (aScores - minAngular) * shapeliness / maxAngular
        eScores = (eScores - minEuclidean) * (1 - shapeliness) / maxEuclidean

        aDict = dict(zip(aIndices, aScores))
        eDict = dict(zip(eIndices, eScores))

        return {key: aDict.get(key, 0) + eDict.get(key, 0) for key in set(aDict) | set(eDict) }