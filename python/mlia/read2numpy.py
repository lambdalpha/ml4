import numpy as np


def read2numpy(filename, sep='\t'):
    with open(filename) as fd:
        lines = fd.readlines()

    return np.array([s.split(sep) for s in filter(lambda x: x != '', [l.strip() for l in lines])], dtype=np.float32)


def test():
    a = read2numpy('../data/datingTestSet.txt')
    print(a)
    print(len(a))
if __name__ == '__main__':
    test()