# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:34:19 2020

@author: 赵怀菩

    这个文件专门放读数据文件的方法
    读数据文件并转换为数组格式
"""
import csv
import numpy as np


def read_iris(filename):
    lis=[]
    with open(filename) as f:
        seq = np.zeros((4, len(f.readlines())-1))
        reader = csv.DictReader(f)
        for row in reader:
            lis.append(row["type"])
    pass

def read_hg19():
    filename = "D:/zhp_workspace/hg18/chr1.fa"
    lis = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i is 0:
                continue
            if i > 5000:
                break
            for ch in line:
                if ch is "A" or ch is "a":
                    lis.append(0)
                elif ch is "G" or ch is "g":
                    lis.append(1)
                elif ch is "C" or ch is "c":
                    lis.append(2)
                elif ch is "T" or ch is "t":
                    lis.append(3)
        # add list into np array
        seq = np.zeros((1, len(lis)))
        for i, tmp in enumerate(lis):
            seq[0][i] = tmp
        print(len(lis),lis)
        #return seq

        return lis #先返回list

if __name__ == "__main__":
    read_iris("D:/zhp_workspace/ucl/iris/iris.data.csv")
    print(read_hg19())