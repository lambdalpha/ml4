import operator

from mlia.Classification import Classification
import numpy as np

class KNN(Classification):
    def __init__(self, debug=False):
        self.debug = debug

    def classify(self, inX, dataset, labels, k):
        """

        :param inX:
        :param dataset: a 2d array n * m
        :param labels:
        :param k:
        """
        dataset_size = dataset.shape[0]
        diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
        dist_mat = np.sqrt(np.square(diff_mat).sum(axis=1))
        sorted_indices = dist_mat.argsort()
        class_count = {}
        for i in range(k):
            # the ith smallest distance label
            vote_I_label = labels[sorted_indices[i]]
            # label count for the k nearest neighbour
            class_count[vote_I_label] = class_count.get(vote_I_label, 0) + 1

        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        if self.debug:
            print("Debug: the class counts of all classes")
            print(sorted_class_count)
        return sorted_class_count[0][0]

    def train(self):
        pass





