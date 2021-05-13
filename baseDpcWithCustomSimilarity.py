import advanced_dpc_v2
import bst
import distance



class baseDpcCustom(advanced_dpc_v2.AdvancedDpcV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.knn_distance_k = kwargs['knn_distance_k']

        self.DistKnn = distance.Dist(dist_func_name=kwargs['dist_func_name'],
                                  record_distance_data=kwargs['record_distance_data']
                                  )
        self.cal_distance_new()
        self.cal_dc()
        self.cal_rou()
        self.cal_delta()

    def cal_distance_new(self):
        """
        主要在cal rou中用，也就是第一次计算完所有点间距离后
        结果放在knn距离词典里
        """
        for i, x in enumerate(self.X):
            # 初始化x点的密度

            now_distance_bst = bst.BST([], self.knn_distance_k, 'min')
            #获得x的邻居信息
            for j, y in enumerate(self.X):
                # 不算自己到自己
                if i == j:
                    continue
                """
                计算 x,y 两点的距离
                """
                # value = distance.distance(x, y)
                value = self.Dist.cal_dist(i, j, x, y)
                """
                计算xy距离
                """
                # x点的密度 加上y点的高斯密度 dc相当于截断距离
                now_distance_bst.try_add(value)
            # 获得k个邻居的距离最大值
            n, p = now_distance_bst.get_rightmost()
            nowMaxDis = n.data
            # 获得k个邻居
            neighborList = []
            for j, y in enumerate(self.X):
                if i == j:
                    continue
                value = self.Dist.cal_dist(i, j, x, y)
                if value < nowMaxDis:
                    neighborList.append(j)
            # 减少重复邻居太近，用后k个邻居（没用，不是按距离找的）
            neighborList = neighborList[-self.knn_distance_k:]
            neighborList.append(i)
            # 计算k个邻居到y的平均距离
            for j, y in enumerate(self.X):
                if i == j:
                    continue
                now_knn_distance = 0
                for neari in neighborList:
                    nearx = self.X[neari]
                    # 不算自己到自己
                    if neari == j:
                        continue
                    """
                    计算 x,y 两点的距离
                    """
                    # value = distance.distance(x, y)
                    value = self.Dist.cal_dist(neari, j, nearx, y)
                    """
                    计算xy距离
                    """
                    now_knn_distance = now_knn_distance + value
                # 平均值为当前点到y的knn距离
                now_knn_distance = now_knn_distance / len(neighborList)
                now_dis = self.Dist.cal_dist(i, j, x, y)
                delta=abs(now_knn_distance-now_dis)
                if now_dis>self.dc:
                    self.DistKnn.update_dist_with_value(i,j,x,y,now_knn_distance)
                elif delta<=0.2*self.dc:
                    self.DistKnn.update_dist_with_value(i, j, x, y, now_dis)
                else:
                    self.DistKnn.update_dist_with_value(i, j, x, y, now_knn_distance)
        self.Dist=self.DistKnn

        pass

    def cal_all_distance(self):
        # self的Dist先存最小维度距离，再存最终距离
        for i, x in enumerate(self.X):
            # 初始化x点的密度
            neighborhood = 0
            for j, y in enumerate(self.X):
                """
                计算 x,y 两点的距离
                """
                # value = distance.distance(x, y)
                value = self.Dist.cal_dist(i, j, x, y)
        pass

    pass
if __name__=="__main__":
    pass