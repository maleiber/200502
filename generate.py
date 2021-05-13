from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn import datasets
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Scatter3D
from pyecharts.faker import Faker

from mpl_toolkits.mplot3d import Axes3D  #
from sklearn import datasets
from sklearn import metrics
from sklearn.decomposition import PCA  # PCA 主成分分析
import re
import math
import dpc_tester

def read_air(path):
    # seg: each line have a label and several attribute. it have none of cluster
    data = []
    label = []
    label_dict = {}
    new_index = 0
    with open(path, "r") as p:
        for line in p.readlines()[1:]:
            temp = []
            # print('line', line)
            pattern = re.split(',', line)
            # print('pattern', pattern)
            candidate_row_list = (6, 10, 14, 18, 22, 26)
            new_pattern = [x for i, x in enumerate(pattern) if i in candidate_row_list]
            for i, x in enumerate(new_pattern):
                # print(i, x)
                if x == "NULL":
                    temp.append(0)
                else:
                    temp.append(float(x))
            data.append(temp)

            label.append(0)

        p.close()
    d = np.zeros((data.__len__(), data[0].__len__()))
    d_width = d.shape[1]
    for i, tmp in enumerate(data):
        for k in range(d_width):
            # print(d[i][k], tmp[k])
            d[i][k] = tmp[k]
    d = StandardScaler().fit_transform(d)
    # print('d is ', d, 'label is ', label, d.shape[0], d.shape[1])
    return d, label


def gen_timeseq_from_data(X, window, stride):
    length, width = X.shape
    iter_time = int(math.floor(length / stride))
    iter_time = max(iter_time - 1, 0) # 修正一次
    # newX = np.zeros((length, width))
    newX = []
    for i in range(iter_time):
        t = []
        for j in range(width):
            t.append(0)
        newX.append(t)
    i = 0
    index = 0

    while True:
        if i + stride >= length or i+window >= length:
            break

        for j in range(width):
            newX[index][j] = [x[j] for x in X[i:i+window]]
        index = index + 1
        i = i + stride
    return newX
    pass

def read_ionosphere(path):
    # seg: each line have a label and 19 attribute. it have 7 kinds of cluster
    data = []
    label = []
    label_dict = {}
    new_index = 0
    with open(path, "r") as p:
        for line in p.readlines():
            temp = []
            #print('line', line)
            pattern = re.split(',', line)
            # print('pattern', pattern)
            for i, x in enumerate(pattern[:-1]):
                # print(i, x)
                temp.append(float(x))
            data.append(temp)
            ch = pattern[-1]
            if ch == 'g\n':
                label.append(0)
            else:
                label.append(1)
            # label.append(int(pattern[-1]) - 1)

        p.close()
    d = np.zeros((data.__len__(), data[0].__len__()))
    d_width = d.shape[1]
    for i, tmp in enumerate(data):
        for k in range(d_width):
            # print(d[i][k], tmp[k])
            d[i][k] = tmp[k]
    d = StandardScaler().fit_transform(d)
    # print('d is ', d, 'label is ', label, d.shape[0], d.shape[1])
    return d, label


def read_seeds(path):
    # seg: each line have a label and 19 attribute. it have 7 kinds of cluster
    data = []
    label = []
    label_dict = {}
    new_index = 0
    with open(path, "r") as p:
        for line in p.readlines():
            temp = []
            # print('line', line)
            pattern = re.split('\t', line)
            # print('pattern', pattern)
            for i, x in enumerate(pattern[:-1]):
                # print(i, x)
                temp.append(float(x))
            data.append(temp)

            label.append(int(pattern[-1]) - 1)

        p.close()
    d = np.zeros((data.__len__(), data[0].__len__()))
    d_width = d.shape[1]
    for i, tmp in enumerate(data):
        for k in range(d_width):
            # print(d[i][k], tmp[k])
            d[i][k] = tmp[k]
    d = StandardScaler().fit_transform(d)
    # print('d is ', d, 'label is ', label, d.shape[0], d.shape[1])
    return d, label

def read_libra_move(path):
    # seg: each line have a label and 19 attribute. it have 7 kinds of cluster
    data = []
    label = []
    label_dict = {}
    new_index = 0
    with open(path, "r") as p:
        for line in p.readlines():
            temp = []
            # print('line', line)
            pattern = re.split(',', line)
            # print('pattern', pattern)
            for i, x in enumerate(pattern[:-1]):
                # print(i, x)
                temp.append(float(x))
            data.append(temp)

            label.append(int(pattern[-1]) - 1)

        p.close()
    d = np.zeros((data.__len__(), data[0].__len__()))
    d_width = d.shape[1]
    for i, tmp in enumerate(data):
        for k in range(d_width):
            # print(d[i][k], tmp[k])
            d[i][k] = tmp[k]
    d = StandardScaler().fit_transform(d)
    # print('d is ', d, 'label is ', label, d.shape[0], d.shape[1])
    return d, label

def read_segmentation(path):
    # seg: each line have a label and 19 attribute. it have 7 kinds of cluster
    data = []
    label = []
    label_dict = {}
    new_index = 0
    with open(path, "r") as p:
        for line in p.readlines():
            temp = []
            # print('line', line)
            pattern = re.split(',', line)
            # print('pattern', pattern)
            for i, x in enumerate(pattern[1:]):
                temp.append(float(x))
            data.append(temp)
            # 记录label的出现，放在label里
            if label_dict.__contains__(pattern[0]):
                pass
            else:
                label_dict[pattern[0]] = new_index
                new_index = new_index + 1
            label.append(label_dict[pattern[0]])

        p.close()
    d = np.zeros((data.__len__(), data[0].__len__()))
    d_width = d.shape[1]
    for i, tmp in enumerate(data):
        for k in range(d_width):
            # print(d[i][k], tmp[k])
            d[i][k] = tmp[k]
    d = StandardScaler().fit_transform(d)
    # print('d is ', d, 'label is ', label, d.shape[0], d.shape[1])
    return d, label

def read_waveform(path):
    # waveform: each line have 21 attribute, and a label
    data = []
    label = []
    with open(path, "r") as p:
        for line in p.readlines():
            temp = []
            # print('line', line)
            pattern = re.split('(-?\d+(.\d*)?)', line)
            # print('pattern', pattern)
            for i, x in enumerate(pattern[:-3]):
                if i % 3 == 1:
                    temp.append(float(x))
            data.append(temp)
            label.append(int(pattern[-3]))
        p.close()
    d = np.zeros((data.__len__(), data[0].__len__()))
    d_width = d.shape[1]
    for i, tmp in enumerate(data):
        for k in range(d_width):
            # print(d[i][k], tmp[k])
            d[i][k] = tmp[k]
    d = StandardScaler().fit_transform(d)
    # print('d is ', d, 'label is ', label)
    return d, label


def show_3d_from_blob6d(fig, X, label, title="PCA", pos=111):
    y = label
    # fig = plt.figure(1, figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    ax = fig.add_subplot(pos, projection='3d')
    X_reduced = PCA(n_components=3).fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=10)

    ax.set_title(title)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    # plt.show()

def show_3d_from_with_sc(fig, X, label, title="PCA", pos=111):
    y = label
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    ax = fig.add_subplot(pos, projection='3d')
    X_reduced = PCA(n_components=3).fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=10)

    if max(label) == 0:
        ax.set_title(title)
    else:
        sc = metrics.silhouette_score(X, label)
        ax.set_title(title+" SC:"+str(sc)[:5])

    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    # plt.show()

def show_result_3d_from_data(fig, d, cluCenter, label, title="PCA", pos=111):
    # 将cluCenter 拆回point和label形式
    dimension_num = d.shape[1]
    # # 这里会有3乘，因为此时每个点是时间窗口。每个点时间跨度为3，所以乘3得到原先数据点位置
    # # vec_list = [[3 * x[0]]+[3 * y for y in x[1]] for x in a.cluCenter]
    vec_list = [[x[0]]+[y for y in x[1]] for x in cluCenter]
    sum_len = sum([len(x) for x in vec_list])
    print_X = np.zeros((sum_len, dimension_num))
    print_label = [0 for x in label]
    temp_index = 0

    for i, x in enumerate(vec_list):
        for j, y in enumerate(x):
            for k in range(dimension_num):
                print_X[temp_index][k] = d[y][k]
            """
            print_X[temp_index][0] = d[y][0]
            print_X[temp_index][1] = d[y][1]
            print_X[temp_index][2] = d[y][2]
            print_X[temp_index][3] = d[y][3]
            print_X[temp_index][4] = d[y][4]
            print_X[temp_index][5] = d[y][5]
            """
            # print_label.append(i)
            print_label[y] = i
            temp_index = temp_index + 1


    X_reduced = PCA(n_components=3).fit_transform(print_X)
    new_print_label = dpc_tester.get_matched_label_from_pred(label, print_label, 3)
    # print('old:',print_label, '\nnew:', new_print_label,'\nreal',label)
    clu_label = dpc_tester.get_matched_label_from_pred(label, new_print_label, 3)
    show_3d_from_with_sc(fig, d, clu_label, title, pos)
    return
    y = print_label
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    ax = fig.add_subplot(pos, projection='3d')
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=10)

    sc = metrics.silhouette_score(print_X, print_label)
    ax.set_title(title+" SC:"+str(sc)[:5])
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    # plt.show()
    return ax


def show_3d_iris():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 指选择第一个和第三个特征作为输入
    y = iris.target  # 输出

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    """
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # 绘制训练点
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')  # 以花瓣长度和宽度为横纵坐标绘制一个图

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    """
    # 为了更好了解维度关系
    # 绘制一个3维的PCA
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2ed eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()


def generate_blob():
    # 生成多类单标签数据集
    center = [[1, 1], [-1, -1], [1, -1]]
    cluster_std = 0.3
    X, labels = make_blobs(n_samples=750, centers=center, n_features=2,
                           cluster_std=0.4, random_state=0)
    X = StandardScaler().fit_transform(X)
    print('X.shape', X.shape)
    print("labels", set(labels))

    unique_lables = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
                 markersize=7)
    plt.title('data by make_blob()')
    plt.show()
    return [X, labels]
    pass


def generate_blobV1():
    # 生成多类单标签数据集
    center = [[1, 1, 1, 1, 1, 1], [1, 0.9, 1, 0.9, 0.9, -1], [1, -1, 1, -1, 1, -1]]
    cluster_std = 0.3
    X, labels = make_blobs(n_samples=750, centers=center, n_features=6,
                           cluster_std=0.4, random_state=0)

    # # 把前2类的 第6维做 随机化
    # X = rand_dimen(X, 5, labels, 0)
    # X = rand_dimen(X, 5, labels, 1)

    X = StandardScaler().fit_transform(X)
    """
    print('X.shape', X.shape)
    print("labels", set(labels))

    unique_lables = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
                 markersize=7)
    plt.title('data by make_blob()')
    plt.show()
    """
    # show_3d_from_blob6d(X, labels)
    return [X, labels]
    pass


def generate_blobV2():
    # 生成多类单标签数据集
    center = [[1, 1, 1, 1, 1, 1], [1, 0.9, 1, 0.9, 0.9, -1], [1, -1, 1, -1, 1, -1], [0, -0.9, 1, -0.9, 1, -0.9]]
    cluster_std = 0.3
    X, labels = make_blobs(n_samples=750, centers=center, n_features=6,
                           cluster_std=0.4, random_state=0)

    # # 把前2类的 第6维做 随机化
    # X = rand_dimen(X, 5, labels, 0)
    # X = rand_dimen(X, 5, labels, 1)
    # # 把后2类的 第1维做 随机化
    # X = rand_dimen(X, 0, labels, 2)
    # X = rand_dimen(X, 0, labels, 3)

    X = StandardScaler().fit_transform(X)
    """
    print('X.shape', X.shape)
    print("labels", set(labels))

    unique_lables = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
                 markersize=7)
    plt.title('data by make_blob()')
    plt.show()
    """

    # show_3d_from_blob6d(X, labels)
    return [X, labels]
    pass


def generate_blob_control_group():
    # 生成多类单标签数据集
    center = [[1, 1, 1, 1, 1, 1], [0.8, 0, 1, 0, 0, -1], [1, -1, 0.8, -1, 1, -1]]
    X, labels = make_blobs(n_samples=750, centers=center, n_features=6,
                           cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)
    """
    print('X.shape', X.shape)
    print("labels", set(labels))

    unique_lables = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
                 markersize=7)
    plt.title('data by make_blob()')
    plt.show()
    """
    # show_3d_from_blob6d(X, labels)
    return [X, labels]
    pass


def generate_blob_control_groupV1():
    # 生成多类单标签数据集
    center = [[1, 1, 1, 1, 1, 1], [0.8, 0, 1, 0, 0, -1], [1, -1, 0.8, -1, 1, -1], [0, 0, 0, 0, 0, 0]]
    X, labels = make_blobs(n_samples=750, centers=center, n_features=6,
                           cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)

    """
    print('X.shape', X.shape)
    print("labels", set(labels))
    unique_lables = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
                 markersize=7)
    plt.title('data by make_blob()')
    plt.show()
    """

    # show_3d_from_blob6d(X, labels)
    return [X, labels]
    pass


def rand_dimen(X, k, label, goal_label):
    # X is numpy 2 dimen array
    for i, x in enumerate(X):
        if label[i] == goal_label:
            X[i][k] = random.random()
    return X
