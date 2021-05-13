import advanced_dpc
import numpy as np


class AdvancedDpcV2(advanced_dpc.AdvancedDpcV1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        聚类细分改进dpc需要的 需要用到的参数
        clu_set 已经做好的聚类 是list 每个元素是一个聚类 有2个单元[0]中心 [1]其它点组成的列表
        cluV2_set 细分好的clu_set
        由于聚类细分也用的聚类方法 所以这部分参数也要提供 大部分参数可以复用
        dcfactor 和 cut_off_gamma_rate 最好不同
        先令这些参数都复用
        """

        self.cluV2_set = []
        self.new_clu_belonging = {}  # 记角标 新聚类角标对应原来角标

    def subdivide_clu_set(self, exec_clu_index_list=[]):
        # 对已有的聚类self.cluCenter进行细分
        # 要求在 已经做好聚类并生成cluCenter后再使用此方法 之前的聚类过程要在外部调用
        # 如果需要对特定的clu进行分类，需要设置exec_clu_index_list，否则默认全部进行子类划分
        if len(exec_clu_index_list) == 0:
            exec_clu_index_list = [i for i, clu in enumerate(self.cluCenter)]

        for i, clu in enumerate(self.cluCenter):
            # 先判断是否对当前clu进行划分
            if i not in exec_clu_index_list:
                # 不需要划分跳到下一步
                # 原先和原先天然对应，直接将这个clu划分到结果中
                self.cluV2_set.append(clu)
                self.new_clu_belonging[self.cluV2_set.__len__()] = i  # 记录位置
                continue
            # 用每个clu 生成子数据集
            sub_index_set = [clu[0]] + clu[1]
            sub_data_set = np.zeros((len(sub_index_set), self.X.shape[1]))
            sub_data_index_dict = {}
            for j, x in enumerate(sub_index_set):
                tmp = self.X[x]
                sub_data_index_dict[j] = x
                for k in range(self.X.shape[1]):
                    sub_data_set[j][k] = tmp[k]

            # print(sub_index_set)

            # 再用子数据集创建子聚类实例 参数全都用现有的，但是 一定是完全划分
            sub_dpc = advanced_dpc.AdvancedDpcV1(X=sub_data_set, dcfactor=self.dcfactor, rouc=self.rouc,
                                                 deltac=self.deltac, gammac=self.gammac, rouo=self.rouo,
                                                 deltao=self.deltao, mission_type='full',
                                                 dist_func_name=self.Dist.dist_func_name,
                                                 record_distance_data=self.Dist.record_distance_data)
            # 然后进行聚类
            sub_clu_num = 2
            sub_dpc.cal_rho_delta_by_rate(0.99, 0.99)  # 没用操作
            sub_dpc.cal_gamma_by_top_n(sub_clu_num)  # 其实无效的
            sub_dpc.set_center_knn_n(self.knn_length)
            sub_dpc.set_cut_off_gamma_rate(self.cut_off_gamma_rate)
            sub_dpc.set_iter_top_n_count(1)  # 默认迭代1次就行
            sub_dpc.set_iter_time(sub_clu_num)  # 1 次最多选几个 可能选不了这么多
            sub_dpc.set_clu_num(sub_clu_num)  # 最多分出几个聚类
            sub_dpc.pick_cluster_consider_mission_typeV1()
            # sub_dpc.pick_cluster_consider_mission_type()
            # 将做好的clu center 放入新的clu set
            for new_clu in sub_dpc.cluCenter:
                temp_clu = new_clu
                temp_clu[0] = sub_data_index_dict[temp_clu[0]]
                temp_clu[1] = [sub_data_index_dict[j] for j in temp_clu[1]]
                self.cluV2_set.append(temp_clu)
                # 再转换为 原来的对应位置

                self.new_clu_belonging[self.cluV2_set.__len__()] = i  # 记录位置

        # 结束

    pass

class AdvancedDpcV2_1(advanced_dpc.AdvancedDpcV1_1):
    """
    不实现重整的聚类细分
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        聚类细分改进dpc需要的 需要用到的参数
        clu_set 已经做好的聚类 是list 每个元素是一个聚类 有2个单元[0]中心 [1]其它点组成的列表
        cluV2_set 细分好的clu_set
        由于聚类细分也用的聚类方法 所以这部分参数也要提供 大部分参数可以复用
        dcfactor 和 cut_off_gamma_rate 最好不同
        先令这些参数都复用
        """

        self.cluV2_set = []
        self.new_clu_belonging = {}  # 记角标 新聚类角标对应原来角标

    def subdivide_clu_set(self, exec_clu_index_list=[]):
        # 对已有的聚类self.cluCenter进行细分
        # 要求在 已经做好聚类并生成cluCenter后再使用此方法 之前的聚类过程要在外部调用
        # 如果需要对特定的clu进行分类，需要设置exec_clu_index_list，否则默认全部进行子类划分
        if len(exec_clu_index_list) == 0:
            exec_clu_index_list = [i for i, clu in enumerate(self.cluCenter)]

        for i, clu in enumerate(self.cluCenter):
            # 先判断是否对当前clu进行划分
            if i not in exec_clu_index_list:
                # 不需要划分跳到下一步
                # 原先和原先天然对应，直接将这个clu划分到结果中
                self.cluV2_set.append(clu)
                self.new_clu_belonging[self.cluV2_set.__len__()] = i  # 记录位置
                continue
            # 用每个clu 生成子数据集
            sub_index_set = [clu[0]] + clu[1]
            sub_data_set = np.zeros((len(sub_index_set), self.X.shape[1]))
            sub_data_index_dict = {}
            for j, x in enumerate(sub_index_set):
                tmp = self.X[x]
                sub_data_index_dict[j] = x
                for k in range(self.X.shape[1]):
                    sub_data_set[j][k] = tmp[k]

            # print(sub_index_set)

            # 再用子数据集创建子聚类实例 参数全都用现有的，但是 一定是完全划分
            sub_dpc = advanced_dpc.AdvancedDpcV1_1(X=sub_data_set, dcfactor=self.dcfactor, rouc=self.rouc,
                                                 deltac=self.deltac, gammac=self.gammac, rouo=self.rouo,
                                                 deltao=self.deltao, mission_type='full',
                                                 dist_func_name=self.Dist.dist_func_name,
                                                 record_distance_data=self.Dist.record_distance_data)
            # 然后进行聚类
            sub_clu_num = 2     # 默认一个聚类模式细分出2个聚类子模式
            sub_dpc.cal_rho_delta_by_rate(0.99, 0.99)  # 没用操作
            sub_dpc.cal_gamma_by_top_n(sub_clu_num)  # 其实无效的
            sub_dpc.set_center_knn_n(self.knn_length)
            sub_dpc.set_cut_off_gamma_rate(self.cut_off_gamma_rate)
            sub_dpc.set_iter_top_n_count(1)  # 默认迭代1次就行
            sub_dpc.set_iter_time(sub_clu_num)  # 1 次最多选几个 可能选不了这么多
            sub_dpc.set_clu_num(sub_clu_num)  # 最多分出几个聚类
            sub_dpc.pick_cluster_consider_mission_typeV1()
            # sub_dpc.pick_cluster_consider_mission_type()
            # 将做好的clu center 放入新的clu set
            for new_clu in sub_dpc.cluCenter:
                temp_clu = new_clu
                temp_clu[0] = sub_data_index_dict[temp_clu[0]]
                temp_clu[1] = [sub_data_index_dict[j] for j in temp_clu[1]]
                self.cluV2_set.append(temp_clu)
                # 再转换为 原来的对应位置

                self.new_clu_belonging[self.cluV2_set.__len__()] = i  # 记录位置

        # 结束

    pass
