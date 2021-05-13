# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:34:12 2019

@author: 赵怀菩
    这里定义了使用KNN思想时将要用到的二叉树
    这是允许数值重复的树，node有一个属性记录count，删除时减到0才会删除。
    使用中序遍历可以获得从小到大的排列。
    树的最右部是整棵树的最大值
"""


class Node:  # 树的节点
    def __init__(self, data):
        self.data = data  # 节点的值
        self.count = 1
        self.lchild = None  # 左子节点
        self.rchild = None  # 右子节点


class BST:
    """
    主要使用的方法:
    初始化方法: 用一个list初始化，不用管元素排列顺序。因为它会挨个insert 放入树中
    get_leftmost(): 获取当前树中最右部的 节点n和父节点p。注意不一定是叶子节点。可能是没有右儿子的节点。
    delete_node(n,p): 在探查最大值大于新计算的距离时，把最大值删除要用到的方法。n,p就是get_leftmost得到的数值
    insert(distance): 将新的距离插入树使用的方法。直接将节点的距离值告知即可。首先先查看树里面有没有重复的节点，有就把count+1
        没有就新建一个count为1的node
    """

    def __init__(self, node_list, max_length, rec_type):  # 创建二叉树
        self.size = len(node_list)
        if len(node_list) >= 1:
            self.root = Node(node_list[0])
            self.maxi = 0
            for data in node_list[1:]:
                self.insert(data)
        self.max_length = max_length
        self.rec_type = rec_type
        self.update_count = 0  # 记录从size填满max_length之后 节点的有效更新次数

    def try_add(self, data):  # 尝试插入data元素
        if self.size == 0:
            self.root = Node(data)
            self.size = self.size + 1
        elif self.size < self.max_length:
            self.insert(data)
            self.size = self.size + 1
        # if 此次添加之后len 大于 index
        else:
            # 其他情况bst已建立 使用bst
            # 获得最右边的节点(最大节点)
            n, p = self.get_rightmost()
            if self.rec_type == 'min':  # 记录最小k队列
                if n.data <= data:
                    # 当前最小距离 大于 bst记录的最大的值 直接跳过
                    pass
                else:
                    # 当前距离 小于 bst记录的最大的值，考虑删除bst最大点
                    # 并添加 当前距离
                    # 一删一增 bst数量不变
                    self.delete_node(n, p)
                    # 删除完之后添加 当前距离
                    self.insert(data)
                    # 增加 更新次数 在有效更新时记录次数
                    self.update_count = self.update_count + 1
            else:  # 记录最大k队列
                print('not implement max knn yet')
                pass  # 尚未实装

    # 二叉树搜索
    def search(self, node, parent, data):  # 二叉树搜索
        if node is None:
            return False, node, parent
        if node.data == data:
            node.count = node.count + 1
            return True, node, parent
        if node.data > data:
            return self.search(node.lchild, node, data)
        else:
            return self.search(node.rchild, node, data)

    # 二叉树插入
    def insert(self, data):
        flag, n, p = self.search(self.root, self.root, data)
        if not flag:
            new_node = Node(data)
            if data > p.data:
                p.rchild = new_node
            else:
                p.lchild = new_node

    def delete_rightmost_node(self):
        n, p = self.get_rightmost()
        self.delete_node(n, p)

    def get_rightmost(self):
        """
        return n,p
        n means now node
        p means parent node
        """
        now_node = self.root
        p = self.root
        while now_node.rchild is not None:
            p = now_node
            now_node = now_node.rchild
        return now_node, p

    def delete_node(self, n, p):
        # 先将节点的计数减一 减完如果大于0 不用管
        n.count = n.count - 1
        if n.count > 0:
            return
        # 计数等于0 再考虑删除
        if n.lchild is None:
            if n == p.lchild:
                p.lchild = n.rchild
            else:
                p.rchild = n.rchild
            del p
        elif n.rchild is None:
            if n == p.lchild:
                p.lchild = n.lchild
            else:
                p.rchild = n.lchild
            del p
        else:  # 左右子树均不为空
            pre = n.rchild
            if pre.lchild is None:
                n.data = pre.data
                n.rchild = pre.rchild
                del pre
            else:
                next = pre.lchild
                while next.lchild is not None:
                    pre = next
                    next = next.lchild
                n.data = next.data
                pre.lchild = next.rchild
                del p

    # 二叉树删除
    def delete(self, root, data):
        flag, n, p = self.search(root, root, data)
        if flag is False:
            print("无该关键字，删除失败")
        else:
            if n.lchild is None:
                if n == p.lchild:
                    p.lchild = n.rchild
                else:
                    p.rchild = n.rchild
                del p
            elif n.rchild is None:
                if n == p.lchild:
                    p.lchild = n.lchild
                else:
                    p.rchild = n.lchild
                del p
            else:  # 左右子树均不为空
                pre = n.rchild
                if pre.lchild is None:
                    n.data = pre.data
                    n.rchild = pre.rchild
                    del pre
                else:
                    next = pre.lchild
                    while next.lchild is not None:
                        pre = next
                        next = next.lchild
                    n.data = next.data
                    pre.lchild = next.rchild
                    del p

    # 先序遍历
    def preOrderTraverse(self, node):
        if node is not None:
            print(node.data)
            self.preOrderTraverse(node.lchild)
            self.preOrderTraverse(node.rchild)

    # 中序遍历
    def inOrderTraverse(self, node):
        if node is not None:
            self.inOrderTraverse(node.lchild)
            print(node.data, node.count)
            self.inOrderTraverse(node.rchild)

    # 中序遍历
    def inOrderTraverseGetAll(self, node,ret_list=[]):
        if node is not None:
            self.inOrderTraverseGetAll(node.lchild,ret_list)
            #print(node.data, node.count)
            for i in range(node.count):
                ret_list.append(node.data)
            self.inOrderTraverseGetAll(node.rchild,ret_list)
        return ret_list

    # 后序遍历 改成了返回node 的data 列表
    def postOrderTraverse(self, node):
        node_list = []
        if node is not None:
            node_list = node_list + self.postOrderTraverse(node.lchild)
            node_list = node_list + self.postOrderTraverse(node.rchild)
            # print(node.data)
            node_list.append(node.data)  # 只返回data的内容
        return node_list


class BstDisData(object):
    def __init__(self, d, i, j):
        """
        这个类的bst的data是记录2个点的距离的
        此时可以让bst专用的data
        d是data信息，在比较时比data
        i j 是 2个点的位置信息，用于后手查记录
        """
        self.i = i
        self.j = j
        self.d = d

    def __gt__(self, other):
        return self.d > other.d

    def __lt__(self, other):
        return self.d < other.d

    def __le__(self, other):
        return self.d <= other.d


if __name__ == '__main__':
    a = [49, 38, 65, 97, 60, 76, 13, 27, 5, 1]
    bst = BST(a)  # 创建二叉查找树
    bst.inOrderTraverse(bst.root)  # 中序遍历
    # print (bst.get_rightmost().data)
    # bst.delete(bst.root, 49)

    bst.inOrderTraverse(bst.root)
