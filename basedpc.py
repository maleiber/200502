# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:54:19 2019

@author: 赵怀菩
"""
from typing import Any

import bst

import baseDpcWithCustomSimilarity
import advanced_dpc_v2
import advanced_dpc

import generate
import distance
import math
import read_data
import numpy as np
import matplotlib.pyplot as plt
import dpc_tester
# 先用sklearn里的数据试试
from sklearn import datasets  # 引入数据集,sklearn包含众多数据集
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from sklearn.neighbors import KNeighborsClassifier  # 利用邻近点方式训练数据
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度
from sklearn.metrics.cluster import adjusted_rand_score  # ri系数
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  #


class BaseDpc(object):
    """
    a basic dpc, will use by l——dpc and g——dpc(succeed)
    base dpc提供了一套完整的聚类过程，但由于它要被继承，而且暂定不会被直接使用
    所以把选择聚类中心的步骤注释掉了。

    200110 ldpc和原来不同，算上gdpc 其实一共有4种新型聚类方法
    第一种：对低维改进的，使用聚类再发现技术的DPC (针对未分类数据)
    第二种：对高维改进的，使用聚类细分技术的DPC (针对已分类数据，可以和聚类再发现同时用)
    第三种：对高维，改进度量方式的DPC
    第四种：聚类整合方法。可以看成是聚类(gdpc?)
    ————————————
    以上方法太多了，先尝试实现前两类。然后争取只使用前两种方法就写出paper发表
    后两种在毕业设计里创新
    ————————————
    ldpc基本就是原样 距离度量可能不一样
    gdpc的距离度量也可能不一样，然后就是序列的取用格式不一
    2点之间的距离是需要多次使用的，而且影响时间空间效率 可以保存下来
    """

    def __init__(self, **kwargs):
        """
        kwargs 是一个字典。 使用字符串找里面的参数，然后赋值
        需要X 是输入数据 是2维numpy数组 第一维是长度，第二维是维数
        需要‘dcfactor’ 0-1 的值代表dc取多少百分比
        需要‘rouc’    代表 inner cluster 的 rou 下界要求
        需要‘deltac’  代表 inner cluster 的 delta 下界要求
        需要‘gammac’  代表 inner cluster 的 gamma 下界要求(rou * delta)。没有要求就设为 rou * delta
        需要‘rouo’    代表 outer cluster 的 rou 上界要求
        需要‘deltao’  代表 outer cluster 的 delta
        需要'mission_type' 分类任务种类 有‘full_classify’和‘limit_classify’. full指将所有数据都分类，此时没有离群点。 limit代表对数据进行有限的分类，此时有些数据没有类别
        """

        self.candidate_set = []
        self.cluCenter = []
        self.rouSet = []
        self.dcfactor = 0
        self.rouc = 0
        self.deltac = 0
        self.gammac = 0
        self.rouo = 0
        self.deltao = 0
        self.dc_index = 0
        self.top_k_center_num = 0
        try:
            self.X = kwargs['X']
            self.dcfactor = kwargs['dcfactor']
            self.rouc = kwargs['rouc']
            self.deltac = kwargs['deltac']
            self.gammac = kwargs['gammac']
            self.rouo = kwargs['rouo']
            self.deltao = kwargs['deltao']

            # 以下为 dist使用的参数
            self.Dist = distance.Dist(dist_func_name=kwargs['dist_func_name'],
                                      record_distance_data=kwargs['record_distance_data']
                                      )

        except KeyError as ae:
            print('key error', ae)
        if kwargs.__contains__('mission_type'):
            self.mission_type = kwargs['mission_type']
        else:
            self.mission_type = 'limit'
        self.dc = 0
        self.distance_bst = None
        self.cal_dc()
        self.cal_rou()
        self.cal_delta()
        # self.pick_cluster()  # 选聚类中心
        # self.pick_outlier()  # 选离群点

    def cal_distance(self):  # deprecate
        """
        计算距离还是单独拿出来算比较好
        算rou，delta，划分聚类的时候都要考虑点之间的距离
        这部分还是反向注入吧，算距离由距离这个类专门处理。dpc类不需要知道这么多
        """
        # 初始化变量
        self.dis_dict = {}
        for i, x in enumerate(self.X):
            # caution that i != j avoid the distance 0
            for j, y in enumerate(self.X[i:]):  # 不在这里避免，这会导致最后一个元素算不了
                # 条件已经默认了，i<=j
                """
                这里计算x和y的距离
                """
                dis = distance.distance(x, y)
                """
                这里计算x和y的距离
                """
                # key 用i,j的str存, 查的时候也这么转化
                self.dis_dict[str(i) + ',' + str(j)] = dis
        pass

    def cal_dc_index(self):
        # 计算dc的位置
        # 调整异常dcfactor的值
        if self.dcfactor <= 0 or self.dcfactor > 1:
            self.dcfactor = 0.02
        # 计算非重复的距离数量 n(n+1)/2
        assume_distance_number = (len(self.X) * (len(self.X) + 1)) / 2
        assume_distance_number = int(assume_distance_number)
        # 计算dc位置 = 距离数量 * dcfactor
        self.dc_index = int(assume_distance_number * self.dcfactor)
        if self.dc_index <= 0:
            print('caution that dc index is 0.\n means that each point is single cluster.')
        elif self.dc_index >= assume_distance_number:
            print('caution that dc index is length of distance number.\n means that each point is in one cluster.')
        pass

    def cal_dc(self):
        """
        计算dc的值 先算dc index
            然后用KNN方法随时保存前 index大小的最小距离值(注意距离不要重复)
            全算完之后取末尾的距离
        """
        # if self.dcfactor<=0 or self.dcfactor>1:
        # self.cal_dc_index()
        # distance_list=sorted(self.distance,key=lambda x:x[1])
        self.cal_dc_index()
        bst_list = []  # 存储距离的队列 bst 用改进的二叉树
        self.distance_bst = bst.BST([], self.dc_index, 'min')
        # 首先 认为xy与yx的距离是一样的
        # 为了避免距离的重复计算，xy与yx只算一次
        # 也就是矩阵只算上三角
        zero_count=0
        for i, x in enumerate(self.X):
            # caution that i != j avoid the distance 0
            for j, y in enumerate(self.X[i:]):  # 不在这里避免，这会导致最后一个元素算不了
                # 跳过2点相同的情况
                if i == i + j:
                    continue
                """
                这里计算x和y的距离
                """
                # 从dis_dict取
                # dis = distance.distance(x, y)
                # dis = self.dis_dict[str(i)+','+str(j)]
                dis = self.Dist.cal_dist(i, i + j, x, y)  # 注意这里的位置是i和i+j两点的位置
                if dis==0:
                    zero_count=zero_count+1
                if zero_count>int(0.9*self.dc_index) and dis==0:
                    # 不要让dc前面老是0，
                    # 设定前k个0值最多只能占一半
                    continue
                """
                这里计算x和y的距离
                """
                """
                这里用knn存储目前第k小的元素
                self.distance_bst = bst.BST(...) 这一行在for循环外面创建
                self.distance_bst.try_add(dis)
                最后
                self.distance_bst.get_rightmost 即可
                """
                self.distance_bst.try_add(dis)
                pass
        # 计算完之后取最大值 rightmost
        n, p = self.distance_bst.get_rightmost()
        self.dc = n.data


    def cal_rou(self):
        # 计算所有点的rou (密度) 放在rou set里
        # enumerate 穷举字段 返回每次的 序号i和元素x
        self.rouSet=[]
        for i, x in enumerate(self.X):
            # 初始化x点的密度
            neighborhood = 0
            for j, y in enumerate(self.X):
                """
                计算 x,y 两点的距离
                """
                # value = distance.distance(x, y)
                value = self.Dist.cal_dist(i, j, x, y)
                """
                计算xy距离
                """
                # x点的密度 加上y点的高斯密度 dc相当于截断距离
                neighborhood = neighborhood + math.exp(-(value / self.dc) * (value / self.dc))
            # i点密度计算完成，放入
            self.rouSet.append([i, neighborhood])

        # 将rouset里的密度标准化 使用preprocessing.scale(X)
        # 先提取密度列表再归一化，再放回
        tmp_rou_set = [x[1] for x in self.rouSet]
        tmp_rou_set = preprocessing.scale(tmp_rou_set)
        for i, x in enumerate(self.rouSet):
            # x[1] = tmp_rou_set[i]
            pass
        pass

    def cal_delta(self):
        # the order of rou delta use the original order in X
        # 按照密度值对rou set的密度排序 升序
        density_set = sorted(self.rouSet, key=lambda x: x[1])
        delta_set = []
        max_delta = 0
        # density set 到 -1 意味着最大密度的点不考虑
        for i, pairx in enumerate(density_set[:-1]):
            # 初始化最小值
            min_delta = 999999
            x = pairx[0]
            for j, pairy in enumerate(density_set[i + 1:]):
                # 从比i点密度大的点里面选取
                # 取密度 pair[1]
                y = pairy[0]
                # 计算2个点距离
                """
                计算 x,y 两点的距离
                """
                # dis = distance.multi_L1(self.X[x], self.X[y])   # 这里的距离尽量别都是0
                # dis = distance.distance(self.X[x], self.X[y])
                dis = self.Dist.cal_dist(x, y, self.X[x], self.X[y])
                """
                计算 x,y 两点的距离
                """
                # value2 = pairy[1]
                # 找比i点密度大的点中 与i点距离的最小值
                if type(dis) is type([]):
                    min_delta = min(min_delta, sum(dis))
                else:
                    min_delta = min(min_delta, dis)
            # min delta 作为i点的距离放入 delta set
            delta_set.append([pairx[0], min_delta])
            # 在距离中记录 最大值
            max_delta = max(max_delta, min_delta)
        # 所有点的距离算出后，最大密度点 ([-1]位置) 其距离按出现过的最大距离算
        delta_set.append([density_set[-1][0], max_delta])

        # 算好所有距离后，记录排序好的 delta set
        self.deltaSet = sorted(delta_set, key=lambda x: x[0])

        # 将deltaset里的密度标准化 使用standardscaler
        # 先提取密度列表再归一化，再放回
        tmp_delta_set = [x[1] for x in self.deltaSet]
        tmp_delta_set = preprocessing.scale(tmp_delta_set)
        for i, x in enumerate(self.deltaSet):
            # x[1] = tmp_delta_set[i]
            pass
        pass

    def pick_center_by_rdg(self):
        candidate_set = {i: i for i, x in enumerate(self.X)}  # 一开始所有点都是候选点 注意是字典不是列表，所以可以pop
        center_set = []
        """
        安排中心
        """
        for pair in self.rouSet:
            i, x = pair
            j, y = self.deltaSet[i]
            # x is rou, y is delta of this point
            if x >= self.rouc and y >= self.deltac:
                # x is center
                center_set.append(i)
                candidate_set.pop(i)
                pass
            elif x * y >= self.gammac:
                # x is center
                center_set.append(i)
                candidate_set.pop(i)
                pass
        return {"center": center_set, "candidate": candidate_set}

    def pick_cluster_consider_mission_type(self):
        """
        考虑分类任务种类的聚类选取
        如果分类种类是full 则每个点属于距离最近的聚类中心
        full代表一次全选了
        否则使用不考虑重复的原始pick cluster方法
        :return:
        """
        if self.mission_type == 'limit':
            self.pick_cluster()
        elif self.mission_type == 'full':
            # 初始化
            """
            先安排中心
            """
            ret_center = self.pick_center_by_rdg()  # 安排过程封装
            candidate_set = list(ret_center["candidate"].values())  # 后续把它转化为了列表计算
            center_set = ret_center["center"]

            """
            给中心安排聚类点
            每个点属于距离最近的聚类中心
            """
            i_pos2index_dict = {}
            # 初始化clu center
            count = 0
            for i in center_set:
                self.cluCenter.append([i, []])
                i_pos2index_dict[i] = count
                count = count + 1
            j = 0
            while j < len(candidate_set):  # 由于是列表每次出队首 进行计算
                # 计算i点与key点的距离

                key = candidate_set[j]
                tmp_dis = []

                for i in center_set:
                    # 为i 点 建立聚类簇
                    """
                    这里计算x和y的距离
                    """
                    # dis = distance.distance(self.X[i], self.X[key])
                    dis = self.Dist.cal_dist(i, key, self.X[i], self.X[key])
                    """
                    这里计算x和y的距离
                    """
                    if type(dis) is type([]):  # 这里默认距离是list时求和作为总距离
                        tmp_dis.append([i, sum(dis)])
                    else:
                        tmp_dis.append([i, dis])
                # 排序选取最近的中心

                tmp_dis = sorted(tmp_dis, key=lambda x: x[1])

                j_belong_which_i, j_dis = tmp_dis[0]
                self.cluCenter[i_pos2index_dict[j_belong_which_i]][1].append(key)
                candidate_set.pop(j)

                # j = j + 1
            pass
        pass

    def pick_cluster(self, considerRepeat=False):
        """
        满足 rouc 和 deltac 要求的
        满足 gammac 要求的
        会被选中成为聚类中心。
        就直接使用self的rou和delta set了
        一个点可以属于多个聚类，只要符合条件.但这样效率慢
        一个点属于一个聚类 效率快点
        """
        # 初始化
        ret_center = self.pick_center_by_rdg()
        candidate_set = list(ret_center["candidate"].values())  # 后续把它转化为了列表计算
        center_set = ret_center["center"]

        """
        先安排中心
        """
        for pair in self.rouSet:
            i, x = pair
            j, y = self.deltaSet[i]
            # x is rou, y is delta of this point
            if x >= self.rouc and y >= self.deltac:
                # x is center
                center_set.append(i)
                candidate_set.pop(i)
                pass
            elif x * y >= self.gammac:
                # x is center
                center_set.append(i)
                candidate_set.pop(i)
                pass
        pass
        """
        给中心安排聚类点
        considerRepeat = true 每个中心都安排一遍 由于离群点要用，所以candidate_set 也会记录
        considerRepeat = false 序列X复制到candidate 安排完的点从candidate去除 下一次不考虑
        pickStyle: 一个倍数 距离小于dc * pickStyle的被划分为一类
        ————建议相似度和距离度量只用一种。在delta用了距离，这里最好也用距离————

        """
        # 需要重新定义candidate_set 把其从dict定义到 list
        # 由于candidate_set outlier可能用所以 把它变成self的
        self.cluCenter = []

        # considerRepeat = false

        if considerRepeat == False:
            for i in center_set:
                # 为i 点 建立聚类簇
                tmpCenter = []
                j = 0
                while j < len(candidate_set):
                    # 计算i点与key点的距离
                    key = candidate_set[j]
                    """
                    计算 x,y 两点的距离 i,key
                    """
                    # dis = distance.distance(self.X[i], self.X[key])
                    dis = self.Dist.cal_dist(i, key, self.X[i], self.X[key])
                    """
                    聚类点划分的判断条件
                      简单方法就是看距离是否小于dc 这样太简单，很多点都没有划分
                      复杂方法: 计算每个聚类中心和所有点的相似度，相似度高于一定程度的划分为一个类。划分是重叠的
                        复杂方法简单形式: O(cn),c~k
                        划分是不重叠的：按照聚类中心的gamma，由高到低进行优先划分，优势(明显)中心先到先得
                        复杂方法复杂形式: O(n)
                    这些不同的判断条件在后续实验中可能要拆开，分别尝试效果。
                        一个是如何划分，一个是划分判断是否重叠
                    """
                    if dis <= self.dc:  # 按照阶段距离dc作为划分依据(是否太小)
                        tmpCenter.append(key)  # 合并key点到i点内
                        # 去除j 不加1
                        candidate_set.pop(j)
                        continue
                    j = j + 1

                self.cluCenter.append([i, tmpCenter])
        else:
            for i in center_set:
                tmpCenter = []
                for j in candidate_set:
                    key = j
                    """
                    计算 x,y 两点的距离 i,key
                    """
                    # dis = distance.distance(self.X[i], self.X[key])
                    dis = self.Dist.cal_dist(i, key, self.X[i], self.X[key])

                    if dis <= self.dc:
                        tmpCenter.append(key)

                self.cluCenter.append([i, tmpCenter])

            pass

        # 到最后聚类簇都被安排好了cluCenter

    def pick_outlier(self):
        """
        满足rouo和deltao要求的
        满足gammao要求的
        会被选择成为离群点

        给离群点安排聚类点
        因为离群点稀疏，接着用candidate_set就行
        默认安排完的点从candidate去除 下一次不考虑

        """
        if len(self.candidate_set) == 0:
            self.pick_cluster()

        center_set = []
        self.candidate_set = {i: 1 for i in self.candidate_set}
        """
        先安排中心
        """
        for pair in self.rouSet:
            i, x = pair
            j, y = self.deltaSet[i]
            if x <= self.rouo and y <= self.deltao:
                # x is center
                center_set.append(i)
                self.candidate_set.pop(i)
                pass

        pass
        """
        给中心安排聚类点
        """
        # 需要重新定义candidate_set 把其从dict定义到 list
        # 由于candidate_set outlier可能用所以 把它变成self的

        self.outlierCenter = []

        # considerRepeat = false
        for i in center_set:
            # 为i 点 建立聚类簇
            tmpCenter = []
            j = 0  # j 是 candidate角标
            while j < len(self.candidate_set):
                # 计算i点与key点的距离
                key = self.candidate_set[j]
                """
                计算 x,y 两点的距离 i,key
                """
                dis = distance.distance(self.X[i], self.X[key])
                """
                计算 x,y 两点的距离
                """
                """
                聚类点划分的判断条件
                  简单方法就是看距离是否小于dc 这样太简单，很多点都没有划分
                """
                if dis <= self.dc:
                    # 合并key点到i点内
                    print("add ", key, "to outlier")
                    tmpCenter.append(key)
                    # 去除j 不加1
                    self.candidate_set.pop(j)
                    continue
                j = j + 1

            self.outlierCenter.append([i, tmpCenter])
        # 到最后聚类簇都被安排好了cluCenter

    def cal_rho_delta_by_rate(self, rho, delta):
        # rho & delta is a factor between 0 and 1
        # which decides the rho c and delta c.
        # caution that if use this function, the gamma is also need to be consider
        # complexity is o(n log n) need to sort
        rho_list = sorted(self.rouSet, key=lambda x: x[1])
        delta_list = sorted(self.deltaSet, key=lambda x: x[1])
        if rho > 1:
            rho = 1
        elif rho < 0:
            rho = 0
        if delta > 1:
            delta = 1
        elif delta < 0:
            delta = 0
        self.rouc = rho_list[
            min(int(len(rho_list) * rho)
                , len(rho_list) - 1)
        ][1]

        self.deltac = delta_list[
            min(int(len(delta_list) * delta)
                , len(delta_list) - 1)
        ][1]

    def cal_gamma_by_top_n(self, n=-1):
        if n != -1:
            self.top_k_center_num = n
            gamma_list = []

            for i, x in enumerate(self.rouSet):
                j, y = self.deltaSet[i]
                gamma_list.append(x[1] * self.deltaSet[i][1])
                pass
            # gamma 标准化(因为不改变相对顺序其实没什么用)
            # gamma_list = preprocessing.scale(gamma_list)
            gamma_list = sorted(gamma_list)

            self.gammac = gamma_list[int(-n)]
            return self.gammac

    def analyze_wrong_list(self, wrong_list, label):
        for i in wrong_list:
            for j, y in enumerate(self.cluCenter):
                if i == y[0]:
                    # 聚类中心分错
                    print('wrong clu center', i, self.X[i], 'rou:', self.rouSet[i], 'delta:', self.deltaSet[i])
                    pass
                for z in y[1]:
                    if i == z:
                        # 聚类里的点分错
                        print('wrong point', i, self.X[i], 'rou:', self.rouSet[i], 'delta:', self.deltaSet[i])
                        print('to clu center:')
                        for n, m in enumerate(self.cluCenter):
                            dis = self.Dist.cal_dist(i, m[0], self.X[i], self.X[m[0]])
                            print('dis to clu[', n, ']:', dis)
                        print('in cluster it belong to ', j, ', but is ', label[i], ' in label')
                        pass

    pass


def use_kmean(fig, k_num, d, label, tit):
    # kmeans
    y_pred = KMeans(n_clusters=k_num, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(fig, d, y_pred, title=tit, pos=232)
    clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,3)
    generate.show_3d_from_with_sc(fig, d, clu_label, title=tit, pos=232)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))
    return accuracy_score(label, clu_label)


def use_dbscan(fig, d, label, tit, e=4):
    # dbscan
    y_pred = DBSCAN(eps=e, min_samples=10).fit_predict(d)
    # generate.show_3d_from_with_sc(fig, d, y_pred, title=tit,pos=233)
    p_min = min(y_pred)
    y_pred = [x - p_min for x in y_pred]
    clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,max(y_pred)+1)
    # print("dbscan:\n",y_pred,"\n")
    generate.show_3d_from_with_sc(fig, d, clu_label, title=tit, pos=233)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    return accuracy_score(label, clu_label)

def use_dbscan_bvdistance(fig, d, label, tit, e=4):
    # dbscan
    y_pred = DBSCAN(eps=e, min_samples=10,metric=distance.BiasVarDistance).fit_predict(d)
    # generate.show_3d_from_with_sc(fig, d, y_pred, title=tit,pos=233)
    p_min = min(y_pred)
    y_pred = [x - p_min for x in y_pred]
    clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,max(y_pred)+1)
    # print("dbscan:\n",y_pred,"\n")
    generate.show_3d_from_with_sc(fig, d, clu_label, title=tit, pos=233)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(d, clu_label))
    return accuracy_score(label, clu_label)


def test_air():
    # air 长2700 左右 维度6 无标签。 所以只能指定一定数量，或者按照百分比聚类，结果无法比较acc等指标
    segmentation = generate.read_air("./co_id2.csv")  # 引入2号大气监测站 数据
    iris_X = segmentation[0]  # 特征变量
    iris_y = segmentation[1]  # 目标值
    label = segmentation[1]
    # d = read_data.read_hg19()[:5000]
    d = segmentation[0]
    print(d.shape[0], d.shape[1])
    d = StandardScaler().fit_transform(d)
    # 先测试生成窗口数据 时间窗口数据
    # test_d = generate.gen_timeseq_from_data(d, 6, 3)
    # print(test_d[0])

    # generate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # kmeans
    # y_pred = KMeans(n_clusters=6, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="Beijing air pollution SC:")

    # dbscan
    # y_pred = DBSCAN(eps=1, min_samples=10).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="Beijing air pollution SC:")
    # return

    # base dpc
    # a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
    #             dist_func_name='multi_L1', record_distance_data=1)
    # a.cal_rho_delta_by_rate(0.99, 0.99)
    # a.cal_gamma_by_top_n(6)
    # a.pick_cluster_consider_mission_type()


    # # 测试改进DPC v1 聚类再发现
    # a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
    #                                mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    # a.cal_rho_delta_by_rate(0.99, 0.99)
    # a.cal_gamma_by_top_n(6)
    # a.set_center_knn_n(6)  # 25
    # a.set_cut_off_gamma_rate(0.9)  # 0.2 0.02
    # a.set_iter_top_n_count(6)
    # a.set_iter_time(6)
    # a.set_clu_num(6)
    # a.pick_cluster_consider_mission_typeV1()
    # # a.pick_cluster()
    # # a.pick_cluster_consider_mission_type()


    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(3)
    a.set_cut_off_gamma_rate(0.9)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # 测试 改进DPC v2 聚类细分
    a.cluCenter = a.cluV2_set
    a.set_clu_num(5)
    a.merge_clu()

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    # tester = dpc_tester.DpcTester(label=label)
    # tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.gen_clu_label_from_clu_center(a.cluV2_set)
    # tester.get_score(d)
    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()

    generate.show_result_3d_from_data(d, a.cluCenter, title="Beijing air pollution SC:")

    # # 打出找到的 cluster 降为3d形式 stride = 3
    # # 这里会有3乘，因为此时每个点是时间窗口。每个点时间跨度为3，所以乘3得到原先数据点位置
    # # vec_list = [[3 * x[0]]+[3 * y for y in x[1]] for x in a.cluCenter]
    # vec_list = [[x[0]]+[y for y in x[1]] for x in a.cluCenter]
    # sum_len = sum([len(x) for x in vec_list])
    #
    # print_X = np.zeros((sum_len, 6))
    # print_label = []
    # temp_index = 0
    #
    # for i, x in enumerate(vec_list):
    #     for j, y in enumerate(x):
    #         print_X[temp_index][0] = d[y][0]
    #         print_X[temp_index][1] = d[y][1]
    #         print_X[temp_index][2] = d[y][2]
    #         print_X[temp_index][3] = d[y][3]
    #         print_X[temp_index][4] = d[y][4]
    #         print_X[temp_index][5] = d[y][5]
    #
    #         print_label.append(i)
    #         temp_index = temp_index + 1
    # X_reduced = PCA(n_components=3).fit_transform(print_X)
    # y = print_label
    # fig = plt.figure(1, figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
    #
    # ax.set_title('beijing air pullotion')
    # ax.set_xlabel("1st eigenvector")
    # ax.w_xaxis.set_ticklabels([])
    # ax.set_ylabel("2nd eigenvector")
    # ax.w_yaxis.set_ticklabels([])
    # ax.set_zlabel("3rd eigenvector")
    # ax.w_zaxis.set_ticklabels([])
    #
    # plt.show()

    # plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    # plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    # plt.show()
    del a

def test_ionosphere():
    # ionosphere 适合 DPCv1 不适合v2 length  350 ,dimen 34, 2 clu
    ###引入数据###
    segmentation = generate.read_ionosphere("./ionosphere.data")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    iris_X = segmentation[0]  # 特征变量
    iris_y = segmentation[1]  # 目标值
    label = segmentation[1]
    # d = read_data.read_hg19()[:5000]
    d = segmentation[0]
    print(d.shape[0], d.shape[1])
    d = StandardScaler().fit_transform(d)

    # generate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # kmeans
    # y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="ionosphere SC:")
    # clu_label = y_pred
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    # dbscan
    # y_pred = DBSCAN(eps=4, min_samples=10).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="ionosphere SC:")
    # clu_label = y_pred
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    """
    # base dpc
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(2)
    a.pick_cluster_consider_mission_type()
    """


    # 测试改进DPC v1 聚类再发现
    # a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
    #                                mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    # a.cal_rho_delta_by_rate(0.99, 0.99)
    # a.cal_gamma_by_top_n(2)
    # a.set_center_knn_n(2) # 25
    # a.set_cut_off_gamma_rate(0.9) # 0.2 0.02
    # a.set_iter_top_n_count(2)
    # a.set_iter_time(2)
    # a.set_clu_num(2)
    # a.pick_cluster_consider_mission_typeV1()
    # # a.pick_cluster()
    # # a.pick_cluster_consider_mission_type()

    """
    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(2)
    a.set_center_knn_n(2)
    a.set_cut_off_gamma_rate(0.9)
    a.set_iter_top_n_count(1)
    a.set_iter_time(2)
    a.set_clu_num(2)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # 测试 改进DPC v2 聚类细分
    a.cluCenter = a.cluV2_set
    a.set_clu_num(2)
    a.merge_clu()
    """
    # print('dc', a.dc)
    # print('rou set', a.rouSet)
    # print('delta set', a.deltaSet)
    # print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    # print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.gen_clu_label_from_clu_center(a.cluV2_set)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_seeds():
    # seeds 适合 DPCv1 不适合v2 length  210, dimen 7, 3 clu
    ###引入数据###
    segmentation = generate.read_seeds("./seeds_dataset.txt")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    iris_X = segmentation[0]  # 特征变量
    iris_y = segmentation[1]  # 目标值
    label = segmentation[1]
    # d = read_data.read_hg19()[:5000]
    d = segmentation[0]
    print(d.shape[0], d.shape[1])
    d = StandardScaler().fit_transform(d)

    # enerate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # kmeans
    # y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="seeds SC:")
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,3)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    # dbscan
    # y_pred = DBSCAN(eps=1, min_samples=10).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="seeds SC:")
    # p_min = min(y_pred)
    # y_pred = [x - p_min for x in y_pred]
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,max(y_pred)+1)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    """
    # base dpc
    a = BaseDpc(X=d, dcfactor=0.1, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.pick_cluster_consider_mission_type()
    """

    """
    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.1, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(3) # 25
    a.set_cut_off_gamma_rate(0.6) # 0.2 0.02
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()
    """

    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.1, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(3)
    a.set_cut_off_gamma_rate(0.6)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # 测试 改进DPC v2 聚类细分
    a.cluCenter = a.cluV2_set
    a.set_clu_num(3)
    a.merge_clu()

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.gen_clu_label_from_clu_center(a.cluV2_set)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_move_libra():
    # libra 适合 DPCv1 不适合v2 length 360 ,dimen 90, 15 clu
    ###引入数据###
    segmentation = generate.read_libra_move("./movement_libras.data")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    iris_X = segmentation[0]  # 特征变量
    iris_y = segmentation[1]  # 目标值
    label = segmentation[1]
    # d = read_data.read_hg19()[:5000]
    d = segmentation[0]
    print(d.shape[0], d.shape[1])
    d = StandardScaler().fit_transform(d)

    generate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    """
    # base dpc
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(15)
    a.pick_cluster_consider_mission_type()
    """


    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(15)
    a.set_center_knn_n(5) # 25
    a.set_cut_off_gamma_rate(0.8) # 0.2 0.02
    a.set_iter_top_n_count(1)
    a.set_iter_time(15)
    a.set_clu_num(15)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()

    """
    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(15)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(15)
    a.set_clu_num(15)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # 测试 改进DPC v2 聚类细分
    a.cluCenter = a.cluV2_set
    a.set_clu_num(15)
    a.merge_clu()
    """
    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.gen_clu_label_from_clu_center(a.cluV2_set)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_segmentation():
    # segmentation 适合 DPCv1 不适合v2 length  2310 dimen 19 7 clu
    ###引入数据###
    segmentation = generate.read_segmentation("./segmentation.txt")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    iris_X = segmentation[0]  # 特征变量
    iris_y = segmentation[1]  # 目标值
    label = segmentation[1]
    # d = read_data.read_hg19()[:5000]
    d = segmentation[0]
    print(d.shape[0], d.shape[1])
    d = StandardScaler().fit_transform(d)

    # generate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # kmeans
    # y_pred = KMeans(n_clusters=7, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="segmentation SC:")
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,7)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    # dbscan
    # y_pred = DBSCAN(eps=1, min_samples=10).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="segmentation SC:")
    # p_min = min(y_pred)
    # y_pred = [x - p_min for x in y_pred]
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,max(y_pred)+1)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    """
    # base dpc
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(19)
    a.pick_cluster_consider_mission_type()
    """
    """
    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(19)
    a.set_center_knn_n(5) # 25
    a.set_cut_off_gamma_rate(0.8) # 0.2 0.02
    a.set_iter_top_n_count(1)
    a.set_iter_time(19)
    a.set_clu_num(19)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()
    """

    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(19)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(19)
    a.set_clu_num(19)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # 测试 改进DPC v2 聚类细分
    a.cluCenter = a.cluV2_set
    a.set_clu_num(19)
    a.merge_clu()

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.gen_clu_label_from_clu_center(a.cluV2_set)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_waveform():
    # waveform适合 DPCv1 不适合v2 length 5000 dimen 21 3 clu
    ###引入数据###
    waveform = generate.read_waveform("./waveform.data")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    iris_X = waveform[0]  # 特征变量
    iris_y = waveform[1]  # 目标值
    label = waveform[1]
    # d = read_data.read_hg19()[:5000]
    d = waveform[0]
    print(d.shape[0], d.shape[1])
    # X = StandardScaler().fit_transform(wine.data)

    # generate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # kmeans
    # y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="waveform SC:")
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,3)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    # dbscan
    # y_pred = DBSCAN(eps=4, min_samples=10).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="waveform SC:")
    # p_min = min(y_pred)
    # y_pred = [x - p_min for x in y_pred]
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,max(y_pred)+1)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    # base dpc
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.pick_cluster_consider_mission_type()

    """
    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()
    """
    """
    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # 测试 改进DPC v2 聚类细分
    a.cluCenter = a.cluV2_set
    a.set_clu_num(3)
    a.merge_clu()
    """
    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.gen_clu_label_from_clu_center(a.cluV2_set)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_wine():
    # wine适合 DPCv1 不适合v2
    ###引入数据###
    wine = datasets.load_wine()  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    iris_X = wine.data  # 特征变量
    iris_y = wine.target  # 目标值
    label = wine.target
    # d = read_data.read_hg19()[:5000]
    d = np.zeros((wine.data.shape[0], wine.data.shape[1]))
    print(d.shape[0], d.shape[1])
    X = StandardScaler().fit_transform(wine.data)
    for i, tmp in enumerate(X):
        # wine have 13 dimen
        d[i][0] = tmp[0]
        d[i][1] = tmp[1]
        d[i][2] = tmp[2]
        d[i][3] = tmp[3]
        d[i][4] = tmp[4]
        d[i][5] = tmp[5]
        d[i][6] = tmp[6]
        d[i][7] = tmp[7]
        d[i][8] = tmp[8]
        d[i][9] = tmp[9]
        d[i][10] = tmp[10]
        d[i][11] = tmp[11]
        d[i][12] = tmp[12]
    d = StandardScaler().fit_transform(d)
    # generate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # # kmeans
    # y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="wine SC:")
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,3)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    # dbscan
    y_pred = DBSCAN(eps=1, min_samples=10).fit_predict(d)
    generate.show_3d_from_with_sc(plt.figure(),d, y_pred, title="wine SC:")
    p_min = min(y_pred)
    y_pred = [x - p_min for x in y_pred]
    clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,max(y_pred)+1)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))
    return
    """
    # base dpc
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.pick_cluster_consider_mission_type()
    """

    """
    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()
    """

    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # 测试 改进DPC v2 聚类细分
    a.cluCenter = a.cluV2_set
    a.set_clu_num(3)
    a.merge_clu()

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.gen_clu_label_from_clu_center(a.cluV2_set)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_iris():
    ###引入数据###
    iris = datasets.load_iris()  # 引入iris鸢尾花数据,iris数据包含4个特征变量
    iris_X = iris.data  # 特征变量
    iris_y = iris.target  # 目标值
    label = iris.target
    # d = read_data.read_hg19()[:5000]
    d = np.zeros((iris.data.shape[0], iris.data.shape[1]))
    X = StandardScaler().fit_transform(iris.data)
    for i, tmp in enumerate(X):
        d[i][0] = tmp[0]
        d[i][1] = tmp[1]
        d[i][2] = tmp[2]
        d[i][3] = tmp[3]
    d = StandardScaler().fit_transform(d)
    # generate.show_3d_from_blob6d(d, label)
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # kmeans
    # y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="iris SC:")
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,3)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    # dbscan
    # y_pred = DBSCAN(eps=1, min_samples=10).fit_predict(d)
    # generate.show_3d_from_with_sc(d, y_pred, title="iris SC:")
    # p_min = min(y_pred)
    # y_pred = [x - p_min for x in y_pred]
    # clu_label = dpc_tester.get_matched_label_from_pred(label,y_pred,max(y_pred)+1)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    # print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    # print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(label, clu_label))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(label, clu_label))
    # print("acc: %0.3f" % accuracy_score(label, clu_label))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(d, clu_label))
    # return
    """
    # base dpc
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.pick_cluster_consider_mission_type()
    """


    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()


    # 测试 改进DPC v2 聚类细分
    # a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
    #                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    # a.cal_rho_delta_by_rate(0.99, 0.99)
    # a.cal_gamma_by_top_n(3)
    # a.set_center_knn_n(5)
    # a.set_cut_off_gamma_rate(0.8)
    # a.set_iter_top_n_count(1)
    # a.set_iter_time(3)
    # a.set_clu_num(3)
    # a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    # a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # # 试试先分再和
    # a.cluCenter = a.cluV2_set
    # a.set_clu_num(3)
    # a.merge_clu()

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))
    
    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def build_one_vector_from_one_cluster(vector, length):
    # 从聚类生成一个向量，标示总样本中属于这类的向量。0不属于，1属于
    ret_vector = [0 for i in range(length)]
    for i, x in enumerate(vector):
        ret_vector[x] = 1
    return ret_vector


def build_n_vector_from_n_cluster(vec_list, length):
    # 样本长度相同 0不属于，1属于
    ret_list = []
    for i, x in enumerate(vec_list):
        ret_list.append(build_one_vector_from_one_cluster(x, length))
    return ret_list


def build_vector_from_DPC_cluster(cluCenter):
    # 返回几个向量代表几个聚类
    ret_vec = []
    for i, one_clu in enumerate(cluCenter):
        tmp_vec = [one_clu[0]]
        tmp_vec = tmp_vec + one_clu[1]
        ret_vec.append(tmp_vec)
    return ret_vec


def build_vector_list_from_label(label, num):
    key = 0  # label from 0 to num
    ret_list = []
    for i in range(num):
        tmp_vec = [0 for x in range(len(label))]
        for j, x in enumerate(label):
            if x == i:
                tmp_vec[j] = 1
        ret_list.append(tmp_vec)
    return ret_list


def get_clu2label_match(vec_list, label_vec_list):
    vec_end = len(vec_list)
    new_list = vec_list + label_vec_list
    new_len = len(new_list)
    match_mtrx = cosine_similarity(new_list)
    # get best match from 0,vec_end to end
    match_pair: Any = []
    match_count = min(len(vec_list), len(label_vec_list))
    for i in range(vec_end):
        tmp_array = []
        for j in range(len(vec_list), new_len):
            tmp_array.append([j, match_mtrx[i][j]])
            tmp_array = sorted(tmp_array, key=lambda x: x[1])
        match_pair.append([i, tmp_array[-1][0] - vec_end])
    return match_pair


def unify_clu_and_label(vec_list, match):
    # 以label中的为准
    length = len(vec_list[0])
    new_label = [-1 for x in range(length)]
    match_dict = {}
    for i, x in enumerate(match):
        match_dict[x[0]] = x[1]

    for i in range(len(vec_list)):
        now_array = vec_list[i]
        for j, y in enumerate(now_array):
            if y == 1:
                new_label[j] = match_dict[i]
    return new_label


def test_blob_v4():
    """
    测试生成4聚类，DPCv2部分
    聚类细分的实例 先分2个，再细分出一个
    """
    d, label = generate.generate_blobV2()
    # d, label = generate.generate_blob_control_groupV1()
    # print(d, [d.shape[1]])
    # d = np.zeros((iris.data.shape[0], iris.data.shape[1]))
    # X = StandardScaler().fit_transform(iris.data)
    X = StandardScaler().fit_transform(d)
    d_width = d.shape[1]
    # print (X)
    for i, tmp in enumerate(X):
        for k in range(d_width):
            d[i][k] = tmp[k]

    # kmeans
    use_kmean(4, d, label, "Ex group:4 cluster")
    # dbscan
    use_dbscan(d,label, "Ex group:4 cluster", e=1)
    return

    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(2)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(2)
    a.set_clu_num(2)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label, 'num:', len(label))

    tester = dpc_tester.DpcTester(label=label)

    # DPCv2 use cluV2_set
    a.cluCenter = a.cluV2_set
    # test when classify 2 cluster
    # a.set_clu_num(2)
    # a.merge_clu()
    # test when classify 3 cluster
    a.set_clu_num(4)
    a.merge_clu()
    tester.gen_clu_label_from_clu_center(a.cluCenter)

    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)

    print('merge 3 clu to 2', a.cluCenter)

    plt.figure()
    generate.show_result_3d_from_data(d, a.cluCenter, title="Beijing air pollution SC:")
    # plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    # plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    # plt.show()
    del a


def test_blob_v3():
    # 测试生成4聚类，DPC和DPCv1部分
    d, label = generate.generate_blobV2()
    # d, label = generate.generate_blob_control_groupV1()
    # print(d, [d.shape[1]])
    # d = np.zeros((iris.data.shape[0], iris.data.shape[1]))
    # X = StandardScaler().fit_transform(iris.data)
    X = StandardScaler().fit_transform(d)
    d_width = d.shape[1]
    # print (X)
    for i, tmp in enumerate(X):
        for k in range(d_width):
            d[i][k] = tmp[k]

    # kmeans
    use_kmean(4, d, label, "Ex group:4 cluster noise")
    # dbscan
    use_dbscan(d,label, "Ex group:4 cluster noise", e=1)
    return
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
    #            dist_func_name='multi_L1', record_distance_data=1)

    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(4)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(4)
    a.set_clu_num(4)
    a.pick_cluster_consider_mission_typeV1()

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)
    a.set_clu_num(2)
    a.merge_clu()
    print('merge 3 clu to 2', a.cluCenter)

    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_blob_v2():
    """
    测试生成3聚类，聚类细分的情况
    聚类细分的实例 先分2个，再细分出一个
    """
    d, label = generate.generate_blobV1()
    # d, label = generate.generate_blob_control_group()
    # print(d, [d.shape[1]])
    # d = np.zeros((iris.data.shape[0], iris.data.shape[1]))
    # X = StandardScaler().fit_transform(iris.data)
    X = StandardScaler().fit_transform(d)
    d_width = d.shape[1]
    # print (X)
    for i, tmp in enumerate(X):
        for k in range(d_width):
            d[i][k] = tmp[k]

    # kmeans
    use_kmean(2, d, label, "Ex group:3 cluster (find 2 motif)")
    # dbscan
    use_dbscan(d, label, "Ex group:3 cluster", e=1)
    return
    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(2)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(2)
    a.set_clu_num(2)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label, 'num:', len(label))

    tester = dpc_tester.DpcTester(label=label)

    # DPCv2 use cluV2_set
    a.cluCenter = a.cluV2_set
    # test when classify 2 cluster
    # a.set_clu_num(2)
    # a.merge_clu()
    # test when classify 3 cluster
    a.set_clu_num(2)
    a.merge_clu()
    tester.gen_clu_label_from_clu_center(a.cluCenter)

    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)

    print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_blob_v2_control_group():
    """
    测试生成4个聚类的对照组，用DPCv2方法
    聚类细分的实例 先分2个，再细分出一个
    """
    d, label = generate.generate_blob_control_groupV1()
    # print(d, [d.shape[1]])
    # d = np.zeros((iris.data.shape[0], iris.data.shape[1]))
    # X = StandardScaler().fit_transform(iris.data)
    X = StandardScaler().fit_transform(d)
    d_width = d.shape[1]
    # print (X)
    for i, tmp in enumerate(X):
        for k in range(d_width):
            d[i][k] = tmp[k]
    # kmeans
    use_kmean(4, d, label, "Control group:4 cluster")
    # dbscan
    use_dbscan(d,label, "Control group:4 cluster", e=1)
    return
    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(2)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(2)
    a.set_clu_num(2)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分

    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)
    a.set_clu_num(2)
    a.merge_clu()
    print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_blob_v1():
    """
    由于输出6合1大图太复杂，
    所以，所有测试都在这里做了
    """
    """
    测试生成3(2)聚类，DPC和DPCv1，DPCv2的情况
    """
    """
    导入3(2)聚类数据集和对照集
    """
    # d, label = generate.generate_blobV1()
    # d, label = generate.generate_blob_control_group()
    """
    导入4聚类数据集 对照集
    """
    # d, label = generate.generate_blobV2()
    # d, label = generate.generate_blob_control_groupV1()
    """
    以下是导入真实数据集
    注意一些数据集已经做了正则化，额外正则化会使效果降低
    """
    # iris 150 3
    iris = datasets.load_iris()  # 引入iris鸢尾花数据,iris数据包含4个特征变量
    iris_X = iris.data  # 特征变量
    iris_y = iris.target  # 目标值
    label = iris.target
    # d = read_data.read_hg19()[:5000]
    d = np.zeros((iris.data.shape[0], iris.data.shape[1]))

    X = StandardScaler().fit_transform(iris.data)

    # # wine 178 3
    # wine = datasets.load_wine()  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    # iris_X = wine.data  # 特征变量
    # iris_y = wine.target  # 目标值
    # label = wine.target
    # # d = read_data.read_hg19()[:5000]
    # d = np.zeros((wine.data.shape[0], wine.data.shape[1]))
    # print(d.shape[0], d.shape[1])
    # # X = StandardScaler().fit_transform(wine.data)
    # X = wine.data

    # # seeds 210 3
    # segmentation = generate.read_seeds("./seeds_dataset.txt")  # 引入seeds
    # iris_X = segmentation[0]  # 特征变量
    # iris_y = segmentation[1]  # 目标值
    # label = segmentation[1]
    # # d = read_data.read_hg19()[:5000]
    # d = segmentation[0]
    # print(d.shape[0], d.shape[1])
    # # X = StandardScaler().fit_transform(d)
    # X = d

    # # ionos 351 2
    # segmentation = generate.read_ionosphere("./ionosphere.data")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    # iris_X = segmentation[0]  # 特征变量
    # iris_y = segmentation[1]  # 目标值
    # label = segmentation[1]
    # # d = read_data.read_hg19()[:5000]
    # d = segmentation[0]
    # print(d.shape[0], d.shape[1])
    # X = StandardScaler().fit_transform(d)
    # # X = d

    # # segment 2310 7
    # segmentation = generate.read_segmentation("./segmentation.txt")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    # iris_X = segmentation[0]  # 特征变量
    # iris_y = segmentation[1]  # 目标值
    # label = segmentation[1]
    # # d = read_data.read_hg19()[:5000]
    # d = segmentation[0]
    # print(d.shape[0], d.shape[1])
    # # X = StandardScaler().fit_transform(d)
    # X = d

    # # waveform 5000 3
    # waveform = generate.read_waveform("./waveform.data")  # 引入wine葡萄酒数据,iris数据包含13个特征变量 3个类
    # iris_X = waveform[0]  # 特征变量
    # iris_y = waveform[1]  # 目标值
    # label = waveform[1]
    # # d = read_data.read_hg19()[:5000]
    # d = waveform[0]
    # print(d.shape[0], d.shape[1])
    # X = d

    """
    以下是导入无标签数据集
    之前无标签的表示方法已经改变，没法再试验了。这里用以前保存的数据
    """
    # segmentation = generate.read_air("./co_id2.csv")  # 引入2号大气监测站 数据
    # iris_X = segmentation[0]  # 特征变量
    # iris_y = segmentation[1]  # 目标值
    # label = segmentation[1]
    # # d = read_data.read_hg19()[:5000]
    # d = segmentation[0]
    # print(d.shape[0], d.shape[1])
    # X = StandardScaler().fit_transform(d)


    # print(d, [d.shape[1]])
    # d = np.zeros((iris.data.shape[0], iris.data.shape[1]))
    # X = StandardScaler().fit_transform(iris.data)

    # X = StandardScaler().fit_transform(d)

    # 输入数据整理到d
    d_width = d.shape[1]
    # print (X)
    try:
        for i in X:
            pass
    except NameError:
        pass
    else:
        for i, tmp in enumerate(X):
            for k in range(d_width):
                # print(k)
                d[i][k] = tmp[k]
                pass

    positive_score = 0
    negative_score = 0

    fig = plt.figure(figsize=plt.figaspect(0.667))

    # here set clu num
    dpc_clu_num = 3
    """
    Here mark the title of data set!
    """
    generate.show_3d_from_with_sc(fig, d, label, title="Iris:"+str(dpc_clu_num)+" cluster",pos= 231)
    """
    Here mark the title of data set!
    """

    """
    第3章部分实验
    """
    """
    # kmeans 对比结果
    negative_score = negative_score + use_kmean(fig, dpc_clu_num, d, label, "K-means")
    # dbscan 对比结果
    negative_score = negative_score + use_dbscan_bvdistance(fig, d, label, "DBSCAN", e=4)
    # 原版dpc 使用曼哈顿距离
    a = BaseDpc(X=d, dcfactor=0.2, rouc=915.0, deltac=990.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.98, 0.99)
    a.cal_gamma_by_top_n(dpc_clu_num)
    a.pick_cluster_consider_mission_type()
    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)

    negative_score = negative_score + tester.get_score(d)
    # plt.figure()
    # plt.subplot(234)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="DPC with manhattan", pos=234)
    # a.pick_cluster()

    # 偏差方差
    a = BaseDpc(X=d, dcfactor=0.2, rouc=915.0, deltac=990.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='BiasVarDistance', record_distance_data=1)

    a.cal_rho_delta_by_rate(0.98, 0.99)
    a.cal_gamma_by_top_n(dpc_clu_num)
    a.pick_cluster_consider_mission_type()
    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)

    negative_score = negative_score + tester.get_score(d)
    # plt.figure()
    # plt.subplot(234)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="DPC with BVDistance", pos=235)
    # a.pick_cluster()

    # 最小n维+knn
    a = baseDpcWithCustomSimilarity.baseDpcCustom(X=d, dcfactor=0.2, rouc=915.0, deltac=990.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='minNDistance', record_distance_data=1, knn_distance_k=3)
    a.cal_rho_delta_by_rate(0.98, 0.99)
    a.cal_gamma_by_top_n(dpc_clu_num)
    a.pick_cluster_consider_mission_type()
    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # print(a.cluCenter)
    negative_score = negative_score + tester.get_score(d)
    # plt.figure()
    # plt.subplot(234)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="DPC with minN&knnDistance", pos=236)
    # a.pick_cluster()
    plt.show()
    """

    """
    第4章部分实验
    """
    # kmeans
    negative_score = negative_score + use_kmean(fig, dpc_clu_num, d, label, "K-means")

    # dbscan
    negative_score = negative_score + use_dbscan(fig, d, label, "DBSCAN", e=4)
    # return
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    a = BaseDpc(X=d, dcfactor=0.2, rouc=915.0, deltac=990.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.98, 0.99)
    a.cal_gamma_by_top_n(dpc_clu_num)
    a.pick_cluster_consider_mission_type()
    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # print(a.cluCenter)
    negative_score = negative_score + tester.get_score(d)
    # plt.figure()
    # plt.subplot(234)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="DPC", pos=234)
    # a.pick_cluster()

    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1_1(X=d, dcfactor=0.2, rouc=915.0, deltac=990.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(dpc_clu_num)  # k clu
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(dpc_clu_num)  # k clu
    a.set_clu_num(dpc_clu_num)  # k clu
    a.pick_cluster_consider_mission_typeV1()

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    positive_score = tester.get_score(d)
    # plt.figure()
    # plt.subplot(235)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="RSDPC(R)", pos=235)
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()

    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2_1(X=d, dcfactor=0.2, rouc=915.0, deltac=990.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(2)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(2)
    a.set_clu_num(2)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    # a.subdivide_clu_set([0])  # 主要是这一步 完成聚类细分
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # DPCv2 use cluV2_set
    a.cluCenter = a.cluV2_set
    # test when classify 2 cluster
    # a.set_clu_num(3)
    # a.merge_clu()
    # test when classify 3 cluster

    a.set_clu_num(dpc_clu_num)
    a.merge_clu()
    tester = dpc_tester.DpcTester(label=label)

    tester.gen_clu_label_from_clu_center(a.cluCenter)
    positive_score = max(positive_score, tester.get_score(d))
    # plt.figure()
    # plt.subplot(236)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="RSDPC(R&S)", pos=236)
    plt.show()
    """
    第5章部分实验
    """
    """
    # kmeans
    negative_score = negative_score + use_kmean(fig, dpc_clu_num, d, label, "K-means")

    # dbscan
    negative_score = negative_score + use_dbscan(fig, d, label, "DBSCAN", e=4)

    # return
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
               dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.98, 0.99)
    a.cal_gamma_by_top_n(dpc_clu_num)
    a.pick_cluster_consider_mission_type()
    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    # print(a.cluCenter)
    negative_score = negative_score + tester.get_score(d)
    # plt.figure()
    # plt.subplot(234)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="DPC",pos=234)
    # a.pick_cluster()

    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(dpc_clu_num)  # k clu
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(dpc_clu_num)  # k clu
    a.set_clu_num(dpc_clu_num)  # k clu
    a.pick_cluster_consider_mission_typeV1()

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    positive_score = tester.get_score(d)
    # plt.figure()
    # plt.subplot(235)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="RSDPC(R)",pos=235)
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()

    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(2)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(2)
    a.set_clu_num(2)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    # a.subdivide_clu_set([0])  # 主要是这一步 完成聚类细分
    a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # DPCv2 use cluV2_set
    a.cluCenter = a.cluV2_set
    # test when classify 2 cluster
    # a.set_clu_num(3)
    # a.merge_clu()
    # test when classify 3 cluster

    a.set_clu_num(dpc_clu_num)
    a.merge_clu()
    tester = dpc_tester.DpcTester(label=label)

    tester.gen_clu_label_from_clu_center(a.cluCenter)
    positive_score = max(positive_score, tester.get_score(d))
    # plt.figure()
    # plt.subplot(236)
    generate.show_result_3d_from_data(fig, d, a.cluCenter, label, title="RSDPC(R&S)",pos=236)

    # a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    # print('dc', a.dc)
    # print('rou set', a.rouSet)
    # print('delta set', a.deltaSet)
    # print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    # print('label', label)

    # tester = dpc_tester.DpcTester(label=label)
    # tester.gen_clu_label_from_clu_center(a.cluCenter)
    # tester.get_score(d)

    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)
    # a.set_clu_num(2)
    # a.merge_clu()
    # print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''

    # generate.show_result_3d_from_data(d, a.cluCenter, title="Ex group:3 cluster noise SC:")

    # plt.figure()
    # plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    # plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    print("score delta: %0.3f" % float(positive_score - negative_score))
    plt.show()
    del a
    """

def test_blob_v1_control_group():
    """
    测试生成3(2)聚类对照组的情况
    3(2)个聚类不相似 有3种DPC方法
    """
    d, label = generate.generate_blob_control_group()
    # print(d, [d.shape[1]])
    # d = np.zeros((iris.data.shape[0], iris.data.shape[1]))
    # X = StandardScaler().fit_transform(iris.data)
    X = StandardScaler().fit_transform(d)
    d_width = d.shape[1]
    # print (X)
    for i, tmp in enumerate(X):
        for k in range(d_width):
            d[i][k] = tmp[k]

    # kmeans
    use_kmean(3, d, label, "Control group:3 cluster")
    # dbscan
    use_dbscan(d,label, "Control group:3 cluster", e=1)
    return
    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    # a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full',
    #            dist_func_name='multi_L1', record_distance_data=1)

    # 测试改进DPC v1 聚类再发现
    a = advanced_dpc.AdvancedDpcV1(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                   mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    # a.pick_cluster_consider_mission_type()
    """
    # 测试 改进DPC v2 聚类细分
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(3)
    a.set_center_knn_n(5)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(3)
    a.set_clu_num(3)
    a.pick_cluster_consider_mission_typeV1()  # 第一次聚类还用V1的方法
    """
    # a.subdivide_clu_set()  # 主要是这一步 完成聚类细分
    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    # print('subdivided clu set', a.cluV2_set, 'num:', len(a.cluV2_set))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)
    a.set_clu_num(2)
    a.merge_clu()
    print('merge 3 clu to 2', a.cluCenter)
    '''
    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 3)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)

    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))

    '''
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()
    del a


def test_blob():
    """
    here test base dpc
    基础的，只测DPC
    目前是测试 4 个和 3（2） 个聚类对照组的情况
    """
    # d, label = generate.generate_blob()
    d, label = generate.generate_blob_control_groupV1()  # 测4
    # d, label = generate.generate_blobV1()  # 测3or2

    # kmeans
    use_kmean(4, d, label, "Ex group:4 cluster noise")
    # dbscan
    use_dbscan(d,label, "Ex group:4 cluster noise", e=1)
    return

    a = BaseDpc(X=d, dcfactor=0.4, rouc=15.0, deltac=0.1, gammac=40.0, rouo=1.0, deltao=1.0, mission_type='full',
                dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.98, 0.99)
    a.cal_gamma_by_top_n(2)
    a.pick_cluster_consider_mission_type()
    # a.pick_cluster()
    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('rhoc', a.rouc, 'deltac', a.deltac)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    a.analyze_wrong_list(tester.wrong_list, label)

    # print('vec list', vec_list)
    # print('label,vec,list', label_vec_list)
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()


def test_digit_v1():
    # 测试手写数据集
    digits = datasets.load_digits()
    digits.keys()
    n_samples, n_features = digits.data.shape
    print((n_samples, n_features))
    label = digits.target
    print(label)
    print(digits.data.shape)
    print(digits.images.shape)

    print(np.all(digits.images.reshape((1797, 64)) == digits.data))

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # 绘制数字：每张图像8*8像素点
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        # 用目标值标记图像
        ax.text(0, 7, str(digits.target[i]))
    plt.show()

    d = np.zeros((digits.data.shape[0], digits.data.shape[1]))
    X = StandardScaler().fit_transform(digits.data)
    for i, tmp in enumerate(X):
        for j in range(digits.data.shape[1]):
            d[i][j] = tmp[j]

    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    a = advanced_dpc_v2.AdvancedDpcV2(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0,
                                      mission_type='full', dist_func_name='multi_L1', record_distance_data=1)
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(10)
    a.set_center_knn_n(12)
    a.set_cut_off_gamma_rate(0.8)
    a.set_iter_top_n_count(1)
    a.set_iter_time(10)
    a.set_clu_num(10)
    a.pick_cluster_consider_mission_typeV1()
    # a.pick_cluster()
    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    print('label', label)

    tester = dpc_tester.DpcTester(label=label)
    tester.gen_clu_label_from_clu_center(a.cluCenter)
    tester.get_score(d)
    # tester.get_wrong_label(d)
    # a.analyze_wrong_list(tester.wrong_list, label)

    # print('vec list', vec_list)
    # print('label,vec,list', label_vec_list)
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.scatter([a.rouSet[i][1] for i in tester.wrong_list], [a.deltaSet[i][1] for i in tester.wrong_list], c='r')
    plt.show()


def test_digit():
    # 测试手写数据集
    digits = datasets.load_digits()
    digits.keys()
    n_samples, n_features = digits.data.shape
    print((n_samples, n_features))
    label = digits.target
    print(label)
    print(digits.data.shape)
    print(digits.images.shape)

    print(np.all(digits.images.reshape((1797, 64)) == digits.data))

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # 绘制数字：每张图像8*8像素点
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        # 用目标值标记图像
        ax.text(0, 7, str(digits.target[i]))
    plt.show()

    d = np.zeros((digits.data.shape[0], digits.data.shape[1]))
    X = StandardScaler().fit_transform(digits.data)
    for i, tmp in enumerate(X):
        for j in range(digits.data.shape[1]):
            d[i][j] = tmp[j]

    # 实际有效的dc是除以2的，因为我这里2点之间的距离算了2遍
    a = BaseDpc(X=d, dcfactor=0.2, rouc=15.0, deltac=0.1, gammac=999.0, rouo=1.0, deltao=1.0, mission_type='full')
    a.cal_rho_delta_by_rate(0.99, 0.99)
    a.cal_gamma_by_top_n(10)
    a.pick_cluster_consider_mission_type()
    # a.pick_cluster()
    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter, 'num:', len(a.cluCenter))
    print('label', label)

    vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(a.cluCenter), len(label))

    label_vec_list = build_vector_list_from_label(label, 10)
    vec2label_match = get_clu2label_match(vec_list, label_vec_list)
    print('match', vec2label_match)
    clu_label = unify_clu_and_label(vec_list, vec2label_match)
    print('clu label', clu_label)
    score = adjusted_rand_score(label, clu_label)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(label, clu_label))
    print("Completeness: %0.3f" % metrics.completeness_score(label, clu_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(label, clu_label))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(label, clu_label))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(label, clu_label))
    print("acc: %0.3f" % accuracy_score(label, clu_label))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(d, clu_label))
    plt.figure()
    plt.scatter([x[1] for x in a.rouSet], [x[1] for x in a.deltaSet])
    plt.show()

    pass


if __name__ == '__main__':
    """d = [49, 38, 65, 97, 60, 76, 13, 2, 5, 1]
    a = BaseDpc(X=d, dcfactor=0.1, rouc=199.0, deltac=130.0, gammac=30.0, rouo=1.0, deltao=1.0)
    a.cal_rho_delta_by_rate(0.8, 0.8)
    a.pick_cluster()
    print('dc', a.dc)
    print('rou set', a.rouSet)
    print('delta set', a.deltaSet)
    print('clu set', a.cluCenter)
    # print('outlier', a.outlierCenter)
    del a"""
    # generate.show_3d_iris()
    # print(generate.read_waveform("./waveform.data"))
    # generate.read_segmentation("./segmentation.txt")
    # generate.read_libra_move("./movement_libras.data")
    # generate.read_seeds("./seeds_dataset.txt")
    # generate.read_ionosphere("./ionosphere.data")
    # generate.read_air("./co_id2.csv")

    # test_air()
    # test_ionosphere()
    # test_seeds()
    # test_move_libra()
    # test_segmentation()
    # test_waveform()
    # test_wine()
    # test_iris()
    """
    所有测试都在test blob v1里面做
    """
    test_blob_v1()
    # test_blob_v2()
    # test_blob_v3()
    # test_blob_v4()
    # test_blob_v1_control_group()
    # test_blob_v2_control_group()
    # test_blob()
    # test_digit_v1()

    #

    pass
