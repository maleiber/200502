# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:03:18 2019

@author: 赵怀菩
    在这里定义了很多 distance 度量方法
    把哪种聚类使用什么度量封装起来，放在前面，统一管理。就像 distance 一样
"""
import numpy as np


class Dist(object):
    def __init__(self, **kwargs):
        """
        这是一个计算距离的类。
        计算时输入position和data。
        他把position当作key保留计算的数据，再再次计算时可以直接用，从而用空间换时间
        """
        if kwargs.__contains__('record_distance_data'):  # 记录是否记录算符哦的距离
            self.record_distance_data = kwargs['record_distance_data']
            if self.record_distance_data is not 0:
                self.record_dist = {}
        else:
            self.record_distance_data = 0
        if kwargs.__contains__('dist_func_name'):  # 计算距离的方法
            self.dist_func_name = kwargs['dist_func_name']
        else:
            self.dist_func_name = 'multi_L1'  # 默认 这个方法
            print("will use default")
        # arrange func dict
        self.func_dict = {'multi_L1': multi_L1,
                          'Ldpc_distance': Ldpc_distance,
                          'L1': L1,
                          'ccmDistance': ccmDistance,
                          'BiasVarDistance':BiasVarDistance,
                          'minNDistance':minNDistance}

    def cal_dist(self, i, j, x, y):
        if self.record_distance_data is not 0:
            key = str(i) + ',' + str(j) if i <= j else str(j) + ',' + str(i)
            if self.record_dist.__contains__(key):
                return self.record_dist[key]
            else:
                self.record_dist[key] = self.func_dict[self.dist_func_name](x, y)  # 这里任务计算 x和y 和 y和x 的距离时一样的
            return self.record_dist[key]
        else:
            return self.record_dist[self.dist_func_name](x, y)
        pass

    def update_dist_with_value(self,i, j, x, y, d):
        # 手动更新两点间距离
        key = str(i) + ',' + str(j) if i <= j else str(j) + ',' + str(i)
        # 不管原来有没有，直接替换
        self.record_dist[key]=d
        return self.record_dist[key]
    pass


def distance(x, y):
    """
    这个 distance 是 base dpc 将使用的距离度量
    """
    return multi_L1(x, y)
    # 换成ccm distance
    # return ccmDistance(x,y)[1] #考虑前2维度


def Ldpc_distance(x, y):
    """
    在 LDPC 中将要使用的距离度量
    """
    return ccmDistance(x, y)


def L1(x, y):
    """
    L1度量方法，最简单的1维欧式距离度量。 1范数
    就是2个数的差值 (绝对值)
    """
    if type(x) == list and type(y) == list:
        return sum([abs(p - y[i]) for i, p in enumerate(x)])
    return abs(x - y)


def multi_L1(x, y):
    """
    在多维数据中使用的L1度量方法
    主要是x和y是list，分别计算 list 对应元素的距离
    """
    temp = 0
    for i in range(len(x)):
        temp = temp + L1(x[i], y[i])
    return temp


def ccmDistance(x, y):
    """
    用于多维数据的ccm距离计算方法
    基于使用L2距离或其他方法
    输入x,y都是np array类型
        其中，一行一维度
    """
    """
    介绍一下np.shape:返回长度为2的list，0为行数，1为列数。
        通过这种方法，获得np array的行或列数
    另外，初始化np矩阵的方法是传入一个 元组，包含长宽
    """
    dis = np.zeros((1, np.shape(x)[0]))  # 代表列数

    """
    后面的代码对应的伪代码
    1.先算x和y每个 对应维度 的距离
    for i in dimension:
        dis[i] = distance(x,y)
    2.对所有距离排序(升序)
    sort(dis)
    初始化new dimension 和 temp dimension (包含k个元素)
    3.第i距离 = 前i元素加权平均，代表考虑了i个维度的距离
    for i in dimension:
        temp_dis[i] += dis[i]
        new_dis[i] = temp_dis[i]/i
    return new_dis
    """
    for i in range(np.shape(x)[0]):
        dis[0][i] = L1(x[i], y[i])
    dis = np.sort(dis, axis=1)  # 按行排序
    new_distance = np.zeros((np.shape(x)[0], 1))
    temp_distance = np.zeros((np.shape(x)[0], 1))
    for i in range(np.shape(x)[0]):
        temp_distance[i][0] = temp_distance[i][0] + dis[0][i]
        new_distance[i][0] = temp_distance[i][0] / (i + 1)

    return new_distance
    pass

def minNDistance(x,y,n=2):
    #使用自适应的方法，n为所有维度的一半
    n=int(len(x)*0.6)
    # 维度过低的处理
    if n==1 and len(x)==2:
        n=2
    elif len(x)==1:
        n=1
    elif len(x)==3:
        n=2

    dimList=[]
    for i in range(len(x)):
        temp = L1(x[i], y[i])
        dimList.append(temp)
    dimList=sorted(dimList)
    temp=0
    for i in dimList[:n]:
        temp=temp+i
    temp=temp/n
    return temp
    pass

def BiasVarDistance(x,y,beta=1):
    beta=float(beta)
    if beta<0:
        beta=-beta
    # 先计算最小维度距离
    mini=float('inf')
    for i in range(len(x)):
        delta=abs(x[i]-y[i])
        if delta<mini:
            mini=delta
    b_distance=mini
    # 在计算平均维度距离
    average=0
    for i in range(len(x)):
        delta=abs(x[i]-y[i])
        average=average+delta
    average=average/len(x)
    v_distance=average

    bv_distance=(1.0/(1+beta**2))*b_distance + (beta**2/(1+beta**2))*v_distance
    return bv_distance
    pass

if __name__ == '__main__':
    test = Dist(record_distance_data=1, dist_func_name='multi_L1')
    print(test.cal_dist(1,23,[22],[33]))