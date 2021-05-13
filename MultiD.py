import numpy as np
import distance


class MultiD(object):
    """
    不用这个了
    直接用numpy的二维矩阵
    """

    def __init__(self):
        self.nameList = {}
        # self.valueList = []

    def __getitem__(self, key):
        if not self.nameList.__contains__(key):
            print("MultiD don't have this key", key)
            # return False
        return self.nameList[key]

    def __setitem__(self, key, value):
        self.nameList[key] = value

    pass


class MultiSeq(object):
    """
    记录输入的维度，时间(长度)
    并用长度和维度初始化多维时间序列seq
    seq的第一个元素是维度，第二个元素是时间
    """
    def __init__(self, dimension, length):
        self.dimension = dimension
        self.length = length
        self.seq = np.zeros((dimension, length))


if __name__ == "__main__":
    a = MultiSeq(2, 3)  # 2维 长度3
    a.seq[0][:] = [1, 2, 3]  # 用法
    print(a.seq)            # seq用法[k,t]
    print(a.seq[:, 0:2])    # 取某时间[:,t]
    print(a.seq[0, :])      # 取某维度[k,:]
    # print(a.seq[0][1])
    print(distance.ccmDistance(a.seq[:, 0:2], a.seq[:, 1:3]))
