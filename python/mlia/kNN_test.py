from mlia.kNN import KNN
from mlia.read2numpy import read2numpy
import numpy as np

def test():
    data = read2numpy('../data/datingTestSet.txt')
    features = data[:, 0:-1]
    labels = data[:, -1]
    train_set = features[0:800]
    train_labels = labels[0:800]

    test_set = features[800:]
    test_labels = labels[800:]
    knn = KNN()
    out_labels = []
    for t in test_set:
        l = knn.classify(t, train_set, train_labels, 10)
        out_labels.append(l)

    print(list(zip(test_labels, out_labels)))
    print(len(test_labels))
    print(sum(np.array(out_labels) == test_labels))


if __name__ == '__main__':
    test()
