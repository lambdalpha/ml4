from mlia.LinearRegression import LinearRegression
from mlia.read2numpy import read2numpy


def test():
    data = read2numpy('../data/ex0.txt')
    features = data[:, 0:-1]
    labels = data[:, -1]
    train_set = features[0:150]
    train_labels = labels[0:150]

    test_set = features[150:]
    test_labels = labels[150:]

    reg = LinearRegression()
    reg.train(train_set, train_labels)

    pred = [reg.predict(x) for x in test_set]
    print(reg.w)
    print(list(zip(pred, test_labels)))
if __name__ == '__main__':
    test()