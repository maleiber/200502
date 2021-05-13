import basedpc
import bst


def get_candidate_from_a_clu(x):
    # inline 从一个clu里获得候选center list
    candidate_x = []
    if len(x) == 2:  # 在没有候选中心，拿这个类的center作为候选
        candidate_x.append(x[0])
    else:
        candidate_x = candidate_x + x[2]  # 有就加上 之前的划分，不存在重复元素，否则要考虑去掉重复的元素
    # candidate_x = list(set(candidate_x))
    return candidate_x


def create_new_clu_from_old_and_dis_list(clu_center, dis_list):
    # 实际一次进行多次合并 不靠谱，这个方法弃用
    # 因为可能涉及多次的合并，有多次index角标移动，索性用旧的创建新的
    #  此时还要考虑未涉及的clu
    center_mark = [0 for x in clu_center]
    for p, x in enumerate(dis_list):
        center_mark[x.i] = 1  # i，j可能多次涉及到
        center_mark[x.j] = 1
    # 先加入新的clu，再补上未涉及的clu
    new_clu_center = []
    new_index = []
    for i, x in enumerate(dis_list):
        m = x.i
        n = x.j
        m_pos = -1
        n_pos = -1
        for pos_r, r in enumerate(new_index):
            if m in r:
                m_pos = pos_r
            elif n in r:
                n_pos = pos_r
        if m_pos > -1 and n_pos > -1:
            new_index[m_pos] = new_index[m_pos] + new_index[n_pos]  # 先改再删
            new_index.pop(n_pos)
            pass
        elif m_pos > -1:
            new_index[m_pos] = new_index[m_pos] + [n]
            pass
        elif n_pos > -1:
            new_index[n_pos] = new_index[n_pos] + [m]
            pass
        else:
            new_index.append([m, n])
            pass

        # 不存在就新建并放入
    # new index 是新的clu角标
    for r in new_index:
        tmp_clu_center = [clu_center[r[0]][0], [], []]  # 使用第一个元素的 第一个中心做中心
        for x in r:
            tmp_clu_center[1] = tmp_clu_center[1] + clu_center[x][1]
            tmp_clu_center[2] = tmp_clu_center[2] + [clu_center[x][0]]

        new_clu_center.append(tmp_clu_center)

    return new_clu_center
    pass


def merge_2_clu_by_index(clu_set, new_clu_set):
    # 将2个clu set按照角标位置合并为一个clu set 返回
    # 他们的大小应该一样, 默认一个clu有2个元素，中心
    ret_clu_set = []
    for i, x in enumerate(clu_set):
        ret_clu_set.append([x[0], x[1] + new_clu_set[i][1]])
    return ret_clu_set


class AdvancedDpcV1(basedpc.BaseDpc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        改进dpc需要的 额外参数
        knn_length          1-
        cut_off_gamma_rate  0.5-1
        iter_time           1-
        iter_top_n_count    1-
        clu_num             1-
        """
        self.generate_dict = {}
        self.knn_length = 5  # 默认knn长度为5
        self.cut_off_gamma = 0
        self.cut_off_gamma_rate = 0
        self.iter_top_n_count = 1  # 每轮选几个聚类中心
        self.iter_time = 1  # 至少迭代一次
        self.clu_num = 0
        self.center_list = []
        self.cluCenter = []

        self.cal_cut_off_gamma()  # 这个时候已经算出了一次所有的rou和delta

    def pick_cluster_consider_mission_typeV1(self):
        """
        if self.mission_type == 'limit':
            self.pick_cluster()
        elif self.mission_type == 'full':
        这里的完全不完全划分只在最后部分起效，而不是一开始
        """
        """
        先初始化，安排初始中心
        """
        # ret_center = self.pick_center_by_rdg()
        # self.candidate_set = ret_center["candidate"]  # 改进的DPC的candidate是关键部分，会多次用到
        # center_set = ret_center["center"]
        # 初始化clu_set
        # clu_set = [[x, []] for x in center_set]  # 这种类似初始化在merge_candidate2center里实现了
        clu_set = []
        center_set = []
        # 初始化
        self.cal_cut_off_gamma()  # 先计算gamma 要根据candidate_set 在init里可能参数不够没算成
        # 初始化candidate
        self.candidate_set = {i: i for i, x in enumerate(self.X)}
        for i in range(self.iter_time):
            # 每一轮的选取聚类中心
            self.cal_gamma_by_top_n()  # 计算下次的top n gamma
            center_set = self.iter_pick_center_by_candidate()  # 计算下一轮的 中心list
            if len(center_set) < 1:
                break  # 没有符合条件的center了 提前终止
            new_clu_set = self.merge_candidate2center(center_set)  # 选取聚类中心 加入到所有聚类中心中去 这个方法已经完成了这部分操作
            clu_set = clu_set + new_clu_set
            # clu_set = merge_2_clu_by_index(clu_set, new_clu_set)    # 加入到所有聚类中心中去
        # 针对已经选好的 clu_set 可能会有识别先后导致的不准确，最后再进行一次重新识别
        clu_set = self.re_sort_exist_clu_set(clu_set)
        # 最后的选取聚类中心
        # self.cluCenter = self.merge_last2center(clu_set)
        self.cluCenter = self.merge_last2centerKNN(clu_set)
        # 选完所有聚类后，将其合并
        self.merge_clu()

    def iter_pick_center_by_candidate(self):
        """
        每轮通过candidate和gamma选择center
        返回的center list 只有中心点的列表 不是clu
        """
        center_set = []  # 包含中心点的列表
        if len(self.candidate_set) < 1:
            # 所有候选都划分完了，没得分
            return center_set
        # 验证 gammac 是否有效
        if self.gammac <= self.cut_off_gamma:
            # invalid gamma
            return center_set
        for i, x in enumerate(self.rouSet):
            tmp_gamma = x[1] * self.deltaSet[i][1]
            if tmp_gamma >= self.gammac:
                if self.candidate_set.__contains__(i):  # 存在 candidate
                    # x is center
                    center_set.append(i)
                    self.candidate_set.pop(i)
        # 选完的center从candidate去除
        return center_set
        pass

    def re_sort_exist_clu_set(self, clu_set):
        """
        对第一轮聚类中的点进行 重新划分
        """
        resort_factor = 0.2
        # print('before resort', clu_set)
        center_set = [x[0] for x in clu_set]
        offset = 0  # 本次变更了 一个类里的几个元素，作为偏移量
        for i, x in enumerate(clu_set):  # x[0] 是聚类中心 i是角标 与j对应
            y_dict = {y: 1 for y in x[1]}
            y_dict_pop_list = []
            for j, y in enumerate(y_dict):  # 对于每个聚类的非聚类中心 (聚类中心都选错就没救了) 的点y
                # print('try', y)
                dis_list = self.cal_all_dis_bet2candi([y], center_set)
                # if dis_list[i] != min(dis_list):  # 宽松要求
                if abs(dis_list[i] - min(dis_list)) > resort_factor * self.dc:
                    k = [k for k, z in enumerate(dis_list) if z == min(dis_list)][0]  # 记录 当前所属与实际不同的情况
                    # print('point', y, 'resort from', i, 'to', k)
                    # print('its dis list is:', dis_list)
                    clu_set[k][1].append(y)  # 把当前点 安排过去
                    y_dict_pop_list.append(y)  # 当前点存在于原来的i类位置j
                    # clu_set[i][1].pop(j)
            for y in y_dict_pop_list:
                y_dict.pop(y)
            clu_set[i][1] = list(y_dict)
        # print('resort finish')
        return clu_set

    def merge_candidate2center(self, center):
        """
        每轮执行的 将候选点self.candidate划分到center里的方法  center 是 list
        只考虑candidate里可能不是聚类中心的点，在cut-off gamma 之下的点
        只进行无冲突的划分 原来有冲突的点可能会因为划走无冲突点 而变为无冲突
        不考虑完不完全
        clu_center = [[x,[]] for x in center]
        for i in candidate
            this_clu_point_list = []
            dis_list = cal(i,center)
            conflict = len(list(filter(lambda x:x<dc,dis_list)))
            if conflict > 1 or conflict ==0: # 有冲突
                continue
            elif conflict ==1 # 正好
                # merge
                j = [x for x in range(len(dis_list)) if dis_list[x] == min(dis_list)][0]  # 找到最小dis所在的角标
                # min(dis_list) 可以是knn 确定的（不太行，因为之前所有点都没被划分）
                clu_center[j][1].append(i)

        """
        # clu_center = []  # 划分的聚类中心
        clu_center = [[x, []] for x in center]
        candidate_pop_list = []  # 由于遍历的时候不能变dict size 在事先先记录一下要pop的 最后在pop
        for i, x in enumerate(self.candidate_set):  # candidate is dict , so x is key
            this_clu_point_list = []
            # 考虑是否小于cut off gamma 小于才能继续
            tmp_gamma = self.rouSet[x][1] * self.deltaSet[x][1]
            if tmp_gamma > self.cut_off_gamma:
                continue

            dis_list = self.cal_all_dis_bet2candi([x], center)  # dis list 是x点到每个center的距离
            conflict = len(list(filter(lambda x: x < self.dc, dis_list)))
            if conflict == 0:  # 有冲突
                # print('no clu in dc:', x)
                continue
            elif conflict > 1:
                pass
                # print('conflict x:', x)
            elif conflict == 1:  # 正好
                # merge
                j = [y for y in range(len(dis_list)) if dis_list[y] == min(dis_list)][0]  # 找到最小dis所在的角标 x到哪个center最近
                # min(dis_list) 可以是knn 确定的（不太行，因为之前所有点都没被划分）
                clu_center[j][1].append(x)  # 把x放到哪个center里
                candidate_pop_list.append(x)    # 记录pop的
                # print('merge ', x, 'to clu center', clu_center[j][0])

                # self.candidate_set.pop(x)  # 把x 从candidate 去除
        # self.candidate_set
        for x in candidate_pop_list:
            self.candidate_set.pop(x)   # 统一pop

        return clu_center  # 返回没有冲突的clu

    def merge_last2center(self, clu_set):
        """
        最后尝试将剩余的点划分到center里
        考虑完全或者不完全的划分
        完全划分：剩下的点每个点划分到距离最近的中心中
        不完全划分：到所有中心的最近距离太大(大于dc 或者2倍dc)，不能划分
        """
        center_set = [x[0] for x in clu_set]

        self.generate_partition_by_clu_set(clu_set)  # 非必要操作 和knn比较用到的

        if self.mission_type == 'full':
            # 完全划分
            candidate_pop_list = []  # 由于遍历的时候不能变dict size 在事先先记录一下要pop的 最后在pop
            for i, x in enumerate(self.candidate_set):
                dis_list = self.cal_all_dis_bet2candi([x], center_set)
                min_dis = min(dis_list)
                j = [i for i, x in enumerate(dis_list) if x == min_dis][0]  # 找到最小dis所在的角标 x到哪个center最近
                # 完全划分：剩下的点每个点划分到距离最近的中心中
                # 节省位置 砍掉print
                # self.find_KNN_index_in_clu_set(x, clu_set)
                """
                print('last merge', x, 'to', j, 'min dis is >dc:', min_dis > self.dc, 'its knn is right:',
                      self.find_KNN_index_in_clu_set(x, clu_set) == j)
                # if self.find_KNN_index_in_clu_set(x, clu_set) != j:
                print('dis list:', dis_list)
                """

                clu_set[j][1].append(x)
                candidate_pop_list.append(x)
            for x in candidate_pop_list:
                self.candidate_set.pop(x)  # 统一pop
        else:
            # 不完全划分
            candidate_pop_list = []  # 由于遍历的时候不能变dict size 在事先先记录一下要pop的 最后在pop
            for i, x in enumerate(self.candidate_set):
                dis_list = self.cal_all_dis_bet2candi([x], center_set)
                j = [y for y in range(len(dis_list)) if dis_list[y] == min(dis_list)][0]  # 找到最小dis所在的角标 x到哪个center最近
                # 不完全划分：到所有中心的最近距离太大(大于dc 或者2倍dc)，不能划分 其它才可以
                min_dis = min(dis_list)
                j = [i for i, x in enumerate(dis_list) if x == min_dis][0]  # 找到最小dis所在的角标 x到哪个center最近
                if min_dis > 2 * self.dc:
                    continue
                else:
                    clu_set[j][1].append(x)
                    candidate_pop_list.append(x)
            for x in candidate_pop_list:
                self.candidate_set.pop(x)  # 统一pop
        return clu_set

    def merge_last2centerKNN(self, clu_set):
        """
        这是将数据 全部划分的方法
        将剩余的点划分到center里
        K近邻方法
        对剩余每个点：
            获取dc范围内最近的k个(full 是最近的k点 limit是dc内的k点) 有归属 点 的列表
            看归属点 需要维护一个dict 随时记录那哪个点被划分到哪个类中
            记录列表里归属最多的类，如果归属数目相同，看哪个最近
            这个点属于记录的类
        """
        center_set = [x[0] for x in clu_set]
        self.generate_partition_by_clu_set(clu_set)
        for i, x in enumerate(self.candidate_set):
            dis_list = self.cal_all_dis_bet2candi([x], center_set)
            use_knn = False
            if_min_in_dis_list2close = self.judge_min_in_dis_list2close(dis_list)
            if min(dis_list) > 1 * self.dc:  # 只在点在dc范围内没有邻居时，使用k近邻 划分
                # 只要dc范围内有邻居有所属就直接划分最近点
                key_index = self.find_KNN_index_in_clu_set(x, clu_set)
                use_knn = True
            elif min(dis_list) > 3 * self.dc and self.mission_type == 'limit':
                # 最小距离过远 在limit的时候不分
                # print(x, 'is far away to its closest center')
                continue
            elif self.judge_min_in_dis_list2close(dis_list):
                """
                如果候选聚类比较相似（点到候选聚类的距离差值小）
                则选取最近距离的聚类比较合适
                """
                key_index = [j for j, y in enumerate(dis_list) if y == min(dis_list)][0]
                knn_index = self.find_KNN_index_in_clu_set(x, clu_set)
                if key_index != knn_index:
                    # print('not knn find nearest index', key_index)
                    # print('compare to knn index:', knn_index)
                    pass
            else:  # 直接适用距离划分
                key_index = self.find_KNN_index_in_clu_set(x, clu_set)
                use_knn = True
            # print('merge last:', x, 'to', key_index)
            # print('use knn:', use_knn, 'dis list', dis_list)
            # print('if min in dis list too close:', self.judge_min_in_dis_list2close(dis_list))
            # max key 是他所属的中心 放进clu set
            clu_set[key_index][1].append(x)

        return clu_set

    def judge_min_in_dis_list2close(self, dis_list, min_order=0):
        """
        judge if element in dis list is too close
        by its minimum and 2rd minimum set by [min_order] and [min_order + 1]
        """
        # first judge if len of dis_list is 1
        if len(dis_list) <= 1:
            return False
        close_factor = 0.2 * self.dc # set close factor
        ordered_dis_list = sorted(dis_list)
        x = ordered_dis_list[min_order]
        y = ordered_dis_list[min_order + 1]
        if abs(x - y) < close_factor:
            return True
        return False

    def generate_partition_by_clu_set(self, clu_set):
        """
        通过clu set 生成一个字典 装有哪个点属于哪个类
        字典放入 self.point_belong 中
        """
        generate_list = []
        for i, x in enumerate(clu_set):
            for y in x[1]:  # y表示 类中的点
                generate_list.append([y, x[0]])  # 类中的点y属于聚类中心x0 不需要修改 设置成元组
        self.generate_dict = {x[0]: x[1] for x in generate_list}

    def find_KNN_index_in_clu_set(self, x, clu_set):
        """
        需要先运行self.generate_partition_by_clu_set(clu_set) 生成划分情况
        """
        merged_list = list(self.generate_dict)  # 已经分类的角标
        center_pos2index_set = {x[0]: i for i, x in enumerate(clu_set)}  # 中心的位置到clu set角标映射
        dis_list = self.cal_all_dis_bet2candi([x], merged_list)
        dis_list = [[merged_list[i], x] for i, x in enumerate(dis_list)]
        dis_list = sorted(dis_list, key=lambda y: y[1])
        if self.mission_type == 'limit':
            # limit 的情况 knn list 记录x点dc内的 几个点 他们所属类的位置 x0位置 x1距离
            knn_list = [self.generate_dict[y[0]] for j, y in enumerate(dis_list) if y[1] < self.dc]
            if len(knn_list) > self.knn_length:
                knn_list = knn_list[:self.knn_length]
        else:
            if len(dis_list) > self.knn_length:
                dis_list = dis_list[:self.knn_length]

            knn_list = [self.generate_dict[y[0]] for j, y in enumerate(dis_list)]

            pass
        # 建立一个记录每个类计数的字典
        count_dict = {x[0]: 0 for x in clu_set}
        # 用knn list 计数 并记录最大值
        max_count = 0
        max_key = 0
        for y in knn_list:
            count_dict[y] = count_dict[y] + 1
            if count_dict[y] > max_count:
                max_count = count_dict[y]
                max_key = y
        # print('knn: dis list:', dis_list, 'count dict:', count_dict)
        return center_pos2index_set[max_key]

    def set_center_knn_n(self, n=5):
        self.knn_length = n
        pass

    def set_iter_top_n_count(self, c):
        if c >= 1:
            self.iter_top_n_count = c
        pass

    def set_iter_time(self, n):
        if n >= 1:
            self.iter_time = n

    def set_cut_off_gamma_rate(self, r):
        # gamma_rate 取值0-1 越大越严格
        self.cut_off_gamma_rate = r
        pass

    def cal_cut_off_gamma(self):
        """
        计算gamma终止条件
        当选出的聚类中心gamma小于终止条件时，这个聚类中心无效
        计算将在第一次算出所有rou delta时进行
        """
        if self.cut_off_gamma_rate <= 0 or 1 < self.cut_off_gamma_rate:
            # 无效
            self.cut_off_gamma = 0
        else:
            cut_off_gamma_index = int(len(self.rouSet) * self.cut_off_gamma_rate)
            self.cut_off_gamma = sorted([self.rouSet[i][1] * self.deltaSet[i][1]
                                         for i, x in enumerate(self.rouSet)])[cut_off_gamma_index]
        return self.cut_off_gamma
        pass

    def cal_gamma_by_top_n(self, n=-1):
        """
        一轮计算n个candidate中gamma最大的点，把他们的位置放list返回
        gamma最大并且要大于cut_off_gamma才能返回
        n = iter_top_n_count
        """
        if n != -1:  # n!=-1 区分原始dpc的计算方法
            return super().cal_gamma_by_top_n(n)

        n = self.iter_top_n_count
        gamma_list = []

        for i, x in enumerate(self.rouSet):
            if self.candidate_set.__contains__(i):  # 只添加 candidate 中的 点 candidate 必须初始化完毕
                gamma_list.append(x[1] * self.deltaSet[i][1])
            pass
        # gamma 标准化(因为不改变相对顺序其实没什么用)
        # gamma_list = preprocessing.scale(gamma_list)
        gamma_list = sorted(gamma_list)

        tmp_gammac = 0
        for i, x in enumerate(gamma_list[-n:]):
            if x <= self.cut_off_gamma:  # cut off 必须计算完毕
                # 不符合条件 继续
                continue
            else:
                tmp_gammac = x  # 符合条件  记录并跳出循环
                break
        self.gammac = tmp_gammac
        return self.gammac
        pass

    def set_clu_num(self, num):
        """
        设定最终要求的聚类数量
        不设定也可以 做出聚类，但是效果没有设定的好
        """
        self.clu_num = num
        pass

    def merge_clu(self):
        """
        合并现有的clu
        从clu里不断地挑2个最相似的，合并为1个
        不断重复此步骤，知道达到clu_num要求

        不断地合并和一次性全合并有什么区别
        如果没有 就记录下来 一次全部合并 完事了
        knn只能记录几个距离，不能确定有几个点。所以不能这么做
        """
        if len(self.center_list) < 1:
            # 第一次合并clu 初始化
            for i, x in enumerate(self.cluCenter):
                self.center_list.append(x[0])
        # 算所有clu_center之间的距离 将距离最短的合并为一个
        # 从此之后cluCenter 有3个元素，0一个是中心，1一个是包含点，2一个是候选中心
        merge_num = len(self.cluCenter) - self.clu_num
        # print(merge_num)
        if merge_num < 1:  # 此时不需要 合并
            return
        clu_min_dis_bst = bst.BST([], merge_num, 'min')

        tmp_clu_center = self.cluCenter
        while len(tmp_clu_center) > self.clu_num:
            min_dis = float("inf")
            min_i = -1
            min_j = -1
            for i, x in enumerate(tmp_clu_center):
                candidate_x = get_candidate_from_a_clu(x)
                for j, y in enumerate(tmp_clu_center[i:]):
                    if i != i + j:
                        candidate_y = get_candidate_from_a_clu(y)
                        dis = self.cal_min_dis_bet2candi(candidate_x,
                                                         candidate_y)  # element in candidate are all position
                        if dis < min_dis:
                            min_dis = dis
                            min_i = i
                            min_j = i + j
            # 1次合并i j 修改i，并把j抬出pop
            # 注意这里的ij不再是角标 为了方便变成了min i和min j
            # i = min_i
            # j = min_j
            # print(min_i, min_j, len(tmp_clu_center))
            tmp_clu_center[min_i] = [tmp_clu_center[min_i][0],
                                 tmp_clu_center[min_i][1] + tmp_clu_center[min_j][1],
                                 get_candidate_from_a_clu(tmp_clu_center[min_i]) + get_candidate_from_a_clu(
                                     tmp_clu_center[min_j])]
            tmp_clu_center.pop(min_j)
        self.cluCenter = tmp_clu_center
        # data类要记录 相关的position
        # dis_data = bst.BstDisData(dis, i, j)  # 这里i j放的是cluCenter的角标
        # clu_min_dis_bst.try_add(dis_data)
        # dis_list = clu_min_dis_bst.postOrderTraverse(clu_min_dis_bst.root)
        # new_clu_center = create_new_clu_from_old_and_dis_list(self.cluCenter, dis_list)  # 获得新的clu center
        # self.cluCenter = new_clu_center

        pass

    def cal_min_dis_bet2candi(self, candidate_x, candidate_y):
        dis = float("inf")
        for i, x in enumerate(candidate_x):
            for j, y in enumerate(candidate_y):
                dis = min(dis, self.Dist.cal_dist(x, y, self.X[x], self.X[y]))
        return dis

    def cal_all_dis_bet2candi(self, candidate_x, candidate_y):
        dis = []
        for i, x in enumerate(candidate_x):
            for j, y in enumerate(candidate_y):
                dis.append(self.Dist.cal_dist(x, y, self.X[x], self.X[y]))
        return dis

    pass


class AdvancedDpcV1_1(basedpc.BaseDpc):
    """
    这是没有聚类重整的聚类再发现
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        改进dpc需要的 额外参数
        knn_length          1-
        cut_off_gamma_rate  0.5-1
        iter_time           1-
        iter_top_n_count    1-
        clu_num             1-
        """
        self.generate_dict = {}
        self.knn_length = 5  # 默认knn长度为5
        self.cut_off_gamma = 0
        self.cut_off_gamma_rate = 0
        self.iter_top_n_count = 1  # 每轮选几个聚类中心
        self.iter_time = 1  # 至少迭代一次
        self.clu_num = 0
        self.center_list = []
        self.cluCenter = []

        self.cal_cut_off_gamma()  # 这个时候已经算出了一次所有的rou和delta

    def pick_cluster_consider_mission_typeV1(self):
        """
        if self.mission_type == 'limit':
            self.pick_cluster()
        elif self.mission_type == 'full':
        这里的完全不完全划分只在最后部分起效，而不是一开始
        """
        """
        先初始化，安排初始中心
        """
        # ret_center = self.pick_center_by_rdg()
        # self.candidate_set = ret_center["candidate"]  # 改进的DPC的candidate是关键部分，会多次用到
        # center_set = ret_center["center"]
        # 初始化clu_set
        # clu_set = [[x, []] for x in center_set]  # 这种类似初始化在merge_candidate2center里实现了
        clu_set = []
        center_set = []
        # 初始化
        self.cal_cut_off_gamma()  # 先计算gamma 要根据candidate_set 在init里可能参数不够没算成
        # 初始化candidate
        self.candidate_set = {i: i for i, x in enumerate(self.X)}
        for i in range(self.iter_time):
            # 每一轮的选取聚类中心
            self.cal_gamma_by_top_n()  # 计算下次的top n gamma
            center_set = self.iter_pick_center_by_candidate()  # 计算下一轮的 中心list
            if len(center_set) < 1:
                break  # 没有符合条件的center了 提前终止
            new_clu_set = self.merge_candidate2center(center_set)  # 选取聚类中心 加入到所有聚类中心中去 这个方法已经完成了这部分操作
            clu_set = clu_set + new_clu_set
            # clu_set = merge_2_clu_by_index(clu_set, new_clu_set)    # 加入到所有聚类中心中去
        # 针对已经选好的 clu_set 可能会有识别先后导致的不准确，最后再进行一次重新识别
        clu_set = self.re_sort_exist_clu_set(clu_set)
        # 最后的选取聚类中心
        self.cluCenter = self.merge_last2center(clu_set)
        # self.cluCenter = self.merge_last2centerKNN(clu_set)
        # 选完所有聚类后，将其合并
        self.merge_clu()

    def iter_pick_center_by_candidate(self):
        """
        每轮通过candidate和gamma选择center
        返回的center list 只有中心点的列表 不是clu
        """
        center_set = []  # 包含中心点的列表
        if len(self.candidate_set) < 1:
            # 所有候选都划分完了，没得分
            return center_set
        # 验证 gammac 是否有效
        if self.gammac <= self.cut_off_gamma:
            # invalid gamma
            return center_set
        for i, x in enumerate(self.rouSet):
            tmp_gamma = x[1] * self.deltaSet[i][1]
            if tmp_gamma >= self.gammac:
                if self.candidate_set.__contains__(i):  # 存在 candidate
                    # x is center
                    center_set.append(i)
                    self.candidate_set.pop(i)
        # 选完的center从candidate去除
        return center_set
        pass

    def re_sort_exist_clu_set(self, clu_set):
        """
        对第一轮聚类中的点进行 重新划分
        """
        resort_factor = 1
        # print('before resort', clu_set)
        center_set = [x[0] for x in clu_set]
        offset = 0  # 本次变更了 一个类里的几个元素，作为偏移量
        for i, x in enumerate(clu_set):  # x[0] 是聚类中心 i是角标 与j对应
            y_dict = {y: 1 for y in x[1]}
            y_dict_pop_list = []
            for j, y in enumerate(y_dict):  # 对于每个聚类的非聚类中心 (聚类中心都选错就没救了) 的点y
                # print('try', y)
                dis_list = self.cal_all_dis_bet2candi([y], center_set)
                # if dis_list[i] != min(dis_list):  # 宽松要求
                if abs(dis_list[i] - min(dis_list)) > resort_factor * self.dc:
                    k = [k for k, z in enumerate(dis_list) if z == min(dis_list)][0]  # 记录 当前所属与实际不同的情况
                    # print('point', y, 'resort from', i, 'to', k)
                    # print('its dis list is:', dis_list)
                    clu_set[k][1].append(y)  # 把当前点 安排过去
                    y_dict_pop_list.append(y)  # 当前点存在于原来的i类位置j
                    # clu_set[i][1].pop(j)
            for y in y_dict_pop_list:
                y_dict.pop(y)
            clu_set[i][1] = list(y_dict)
        # print('resort finish')
        return clu_set

    def merge_candidate2center(self, center):
        """
        每轮执行的 将候选点self.candidate划分到center里的方法  center 是 list
        只考虑candidate里可能不是聚类中心的点，在cut-off gamma 之下的点
        只进行无冲突的划分 原来有冲突的点可能会因为划走无冲突点 而变为无冲突
        不考虑完不完全
        clu_center = [[x,[]] for x in center]
        for i in candidate
            this_clu_point_list = []
            dis_list = cal(i,center)
            conflict = len(list(filter(lambda x:x<dc,dis_list)))
            if conflict > 1 or conflict ==0: # 有冲突
                continue
            elif conflict ==1 # 正好
                # merge
                j = [x for x in range(len(dis_list)) if dis_list[x] == min(dis_list)][0]  # 找到最小dis所在的角标
                # min(dis_list) 可以是knn 确定的（不太行，因为之前所有点都没被划分）
                clu_center[j][1].append(i)

        """
        # clu_center = []  # 划分的聚类中心
        clu_center = [[x, []] for x in center]
        candidate_pop_list = []  # 由于遍历的时候不能变dict size 在事先先记录一下要pop的 最后在pop
        for i, x in enumerate(self.candidate_set):  # candidate is dict , so x is key
            this_clu_point_list = []
            # 考虑是否小于cut off gamma 小于才能继续
            tmp_gamma = self.rouSet[x][1] * self.deltaSet[x][1]
            if tmp_gamma > self.cut_off_gamma:
                continue

            dis_list = self.cal_all_dis_bet2candi([x], center)  # dis list 是x点到每个center的距离
            conflict = len(list(filter(lambda x: x < self.dc, dis_list)))
            if conflict == 0:  # 有冲突
                # print('no clu in dc:', x)
                continue
            elif conflict > 1:
                pass
                # print('conflict x:', x)
            elif conflict == 1:  # 正好
                # merge
                j = [y for y in range(len(dis_list)) if dis_list[y] == min(dis_list)][0]  # 找到最小dis所在的角标 x到哪个center最近
                # min(dis_list) 可以是knn 确定的（不太行，因为之前所有点都没被划分）
                clu_center[j][1].append(x)  # 把x放到哪个center里
                candidate_pop_list.append(x)    # 记录pop的
                # print('merge ', x, 'to clu center', clu_center[j][0])

                # self.candidate_set.pop(x)  # 把x 从candidate 去除
        # self.candidate_set
        for x in candidate_pop_list:
            self.candidate_set.pop(x)   # 统一pop

        return clu_center  # 返回没有冲突的clu

    def merge_last2center(self, clu_set):
        """
        最后尝试将剩余的点划分到center里
        考虑完全或者不完全的划分
        完全划分：剩下的点每个点划分到距离最近的中心中
        不完全划分：到所有中心的最近距离太大(大于dc 或者2倍dc)，不能划分
        """
        center_set = [x[0] for x in clu_set]

        self.generate_partition_by_clu_set(clu_set)  # 非必要操作 和knn比较用到的

        if self.mission_type == 'full':
            # 完全划分
            candidate_pop_list = []  # 由于遍历的时候不能变dict size 在事先先记录一下要pop的 最后在pop
            for i, x in enumerate(self.candidate_set):
                dis_list = self.cal_all_dis_bet2candi([x], center_set)
                min_dis = min(dis_list)
                j = [i for i, x in enumerate(dis_list) if x == min_dis][0]  # 找到最小dis所在的角标 x到哪个center最近
                # 完全划分：剩下的点每个点划分到距离最近的中心中
                # 节省位置 砍掉print
                # self.find_KNN_index_in_clu_set(x, clu_set)
                """
                print('last merge', x, 'to', j, 'min dis is >dc:', min_dis > self.dc, 'its knn is right:',
                      self.find_KNN_index_in_clu_set(x, clu_set) == j)
                # if self.find_KNN_index_in_clu_set(x, clu_set) != j:
                print('dis list:', dis_list)
                """

                clu_set[j][1].append(x)
                candidate_pop_list.append(x)
            for x in candidate_pop_list:
                self.candidate_set.pop(x)  # 统一pop
        else:
            # 不完全划分
            candidate_pop_list = []  # 由于遍历的时候不能变dict size 在事先先记录一下要pop的 最后在pop
            for i, x in enumerate(self.candidate_set):
                dis_list = self.cal_all_dis_bet2candi([x], center_set)
                j = [y for y in range(len(dis_list)) if dis_list[y] == min(dis_list)][0]  # 找到最小dis所在的角标 x到哪个center最近
                # 不完全划分：到所有中心的最近距离太大(大于dc 或者2倍dc)，不能划分 其它才可以
                min_dis = min(dis_list)
                j = [i for i, x in enumerate(dis_list) if x == min_dis][0]  # 找到最小dis所在的角标 x到哪个center最近
                if min_dis > 2 * self.dc:
                    continue
                else:
                    clu_set[j][1].append(x)
                    candidate_pop_list.append(x)
            for x in candidate_pop_list:
                self.candidate_set.pop(x)  # 统一pop
        return clu_set

    def merge_last2centerKNN(self, clu_set):
        """
        这是将数据 全部划分的方法
        将剩余的点划分到center里
        K近邻方法
        对剩余每个点：
            获取dc范围内最近的k个(full 是最近的k点 limit是dc内的k点) 有归属 点 的列表
            看归属点 需要维护一个dict 随时记录那哪个点被划分到哪个类中
            记录列表里归属最多的类，如果归属数目相同，看哪个最近
            这个点属于记录的类
        """
        center_set = [x[0] for x in clu_set]
        self.generate_partition_by_clu_set(clu_set)
        for i, x in enumerate(self.candidate_set):
            dis_list = self.cal_all_dis_bet2candi([x], center_set)
            use_knn = False
            if_min_in_dis_list2close = self.judge_min_in_dis_list2close(dis_list)
            if min(dis_list) > 1 * self.dc:  # 只在点在dc范围内没有邻居时，使用k近邻 划分
                # 只要dc范围内有邻居有所属就直接划分最近点
                key_index = self.find_KNN_index_in_clu_set(x, clu_set)
                use_knn = True
            elif min(dis_list) > 3 * self.dc and self.mission_type == 'limit':
                # 最小距离过远 在limit的时候不分
                # print(x, 'is far away to its closest center')
                continue
            elif self.judge_min_in_dis_list2close(dis_list):
                """
                如果候选聚类比较相似（点到候选聚类的距离差值小）
                则选取最近距离的聚类比较合适
                """
                key_index = [j for j, y in enumerate(dis_list) if y == min(dis_list)][0]
                knn_index = self.find_KNN_index_in_clu_set(x, clu_set)
                if key_index != knn_index:
                    # print('not knn find nearest index', key_index)
                    # print('compare to knn index:', knn_index)
                    pass
            else:  # 直接适用距离划分
                key_index = self.find_KNN_index_in_clu_set(x, clu_set)
                use_knn = True
            # print('merge last:', x, 'to', key_index)
            # print('use knn:', use_knn, 'dis list', dis_list)
            # print('if min in dis list too close:', self.judge_min_in_dis_list2close(dis_list))
            # max key 是他所属的中心 放进clu set
            clu_set[key_index][1].append(x)

        return clu_set

    def judge_min_in_dis_list2close(self, dis_list, min_order=0):
        """
        judge if element in dis list is too close
        by its minimum and 2rd minimum set by [min_order] and [min_order + 1]
        """
        # first judge if len of dis_list is 1
        if len(dis_list) <= 1:
            return False
        close_factor = 0.2 * self.dc # set close factor
        ordered_dis_list = sorted(dis_list)
        x = ordered_dis_list[min_order]
        y = ordered_dis_list[min_order + 1]
        if abs(x - y) < close_factor:
            return True
        return False

    def generate_partition_by_clu_set(self, clu_set):
        """
        通过clu set 生成一个字典 装有哪个点属于哪个类
        字典放入 self.point_belong 中
        """
        generate_list = []
        for i, x in enumerate(clu_set):
            for y in x[1]:  # y表示 类中的点
                generate_list.append([y, x[0]])  # 类中的点y属于聚类中心x0 不需要修改 设置成元组
        self.generate_dict = {x[0]: x[1] for x in generate_list}

    def find_KNN_index_in_clu_set(self, x, clu_set):
        """
        需要先运行self.generate_partition_by_clu_set(clu_set) 生成划分情况
        """
        merged_list = list(self.generate_dict)  # 已经分类的角标
        center_pos2index_set = {x[0]: i for i, x in enumerate(clu_set)}  # 中心的位置到clu set角标映射
        dis_list = self.cal_all_dis_bet2candi([x], merged_list)
        dis_list = [[merged_list[i], x] for i, x in enumerate(dis_list)]
        dis_list = sorted(dis_list, key=lambda y: y[1])
        if self.mission_type == 'limit':
            # limit 的情况 knn list 记录x点dc内的 几个点 他们所属类的位置 x0位置 x1距离
            knn_list = [self.generate_dict[y[0]] for j, y in enumerate(dis_list) if y[1] < self.dc]
            if len(knn_list) > self.knn_length:
                knn_list = knn_list[:self.knn_length]
        else:
            if len(dis_list) > self.knn_length:
                dis_list = dis_list[:self.knn_length]

            knn_list = [self.generate_dict[y[0]] for j, y in enumerate(dis_list)]

            pass
        # 建立一个记录每个类计数的字典
        count_dict = {x[0]: 0 for x in clu_set}
        # 用knn list 计数 并记录最大值
        max_count = 0
        max_key = 0
        for y in knn_list:
            count_dict[y] = count_dict[y] + 1
            if count_dict[y] > max_count:
                max_count = count_dict[y]
                max_key = y
        # print('knn: dis list:', dis_list, 'count dict:', count_dict)
        return center_pos2index_set[max_key]

    def set_center_knn_n(self, n=5):
        self.knn_length = n
        pass

    def set_iter_top_n_count(self, c):
        if c >= 1:
            self.iter_top_n_count = c
        pass

    def set_iter_time(self, n):
        if n >= 1:
            self.iter_time = n

    def set_cut_off_gamma_rate(self, r):
        # gamma_rate 取值0-1 越大越严格
        self.cut_off_gamma_rate = r
        pass

    def cal_cut_off_gamma(self):
        """
        计算gamma终止条件
        当选出的聚类中心gamma小于终止条件时，这个聚类中心无效
        计算将在第一次算出所有rou delta时进行
        """
        if self.cut_off_gamma_rate <= 0 or 1 < self.cut_off_gamma_rate:
            # 无效
            self.cut_off_gamma = 0
        else:
            cut_off_gamma_index = int(len(self.rouSet) * self.cut_off_gamma_rate)
            self.cut_off_gamma = sorted([self.rouSet[i][1] * self.deltaSet[i][1]
                                         for i, x in enumerate(self.rouSet)])[cut_off_gamma_index]
        return self.cut_off_gamma
        pass

    def cal_gamma_by_top_n(self, n=-1):
        """
        一轮计算n个candidate中gamma最大的点，把他们的位置放list返回
        gamma最大并且要大于cut_off_gamma才能返回
        n = iter_top_n_count
        """
        if n != -1:  # n!=-1 区分原始dpc的计算方法
            return super().cal_gamma_by_top_n(n)

        n = self.iter_top_n_count
        gamma_list = []

        for i, x in enumerate(self.rouSet):
            if self.candidate_set.__contains__(i):  # 只添加 candidate 中的 点 candidate 必须初始化完毕
                gamma_list.append(x[1] * self.deltaSet[i][1])
            pass
        # gamma 标准化(因为不改变相对顺序其实没什么用)
        # gamma_list = preprocessing.scale(gamma_list)
        gamma_list = sorted(gamma_list)

        tmp_gammac = 0
        for i, x in enumerate(gamma_list[-n:]):
            if x <= self.cut_off_gamma:  # cut off 必须计算完毕
                # 不符合条件 继续
                continue
            else:
                tmp_gammac = x  # 符合条件  记录并跳出循环
                break
        self.gammac = tmp_gammac
        return self.gammac
        pass

    def set_clu_num(self, num):
        """
        设定最终要求的聚类数量
        不设定也可以 做出聚类，但是效果没有设定的好
        """
        self.clu_num = num
        pass

    def merge_clu(self):
        """
        合并现有的clu
        从clu里不断地挑2个最相似的，合并为1个
        不断重复此步骤，知道达到clu_num要求

        不断地合并和一次性全合并有什么区别
        如果没有 就记录下来 一次全部合并 完事了
        knn只能记录几个距离，不能确定有几个点。所以不能这么做
        """
        if len(self.center_list) < 1:
            # 第一次合并clu 初始化
            for i, x in enumerate(self.cluCenter):
                self.center_list.append(x[0])
        # 算所有clu_center之间的距离 将距离最短的合并为一个
        # 从此之后cluCenter 有3个元素，0一个是中心，1一个是包含点，2一个是候选中心
        merge_num = len(self.cluCenter) - self.clu_num
        # print(merge_num)
        if merge_num < 1:  # 此时不需要 合并
            return
        clu_min_dis_bst = bst.BST([], merge_num, 'min')

        tmp_clu_center = self.cluCenter
        while len(tmp_clu_center) > self.clu_num:
            min_dis = float("inf")
            min_i = -1
            min_j = -1
            for i, x in enumerate(tmp_clu_center):
                candidate_x = get_candidate_from_a_clu(x)
                for j, y in enumerate(tmp_clu_center[i:]):
                    if i != i + j:
                        candidate_y = get_candidate_from_a_clu(y)
                        dis = self.cal_min_dis_bet2candi(candidate_x,
                                                         candidate_y)  # element in candidate are all position
                        if dis < min_dis:
                            min_dis = dis
                            min_i = i
                            min_j = i + j
            # 1次合并i j 修改i，并把j抬出pop
            # 注意这里的ij不再是角标 为了方便变成了min i和min j
            # i = min_i
            # j = min_j
            # print(min_i, min_j, len(tmp_clu_center))
            tmp_clu_center[min_i] = [tmp_clu_center[min_i][0],
                                 tmp_clu_center[min_i][1] + tmp_clu_center[min_j][1],
                                 get_candidate_from_a_clu(tmp_clu_center[min_i]) + get_candidate_from_a_clu(
                                     tmp_clu_center[min_j])]
            tmp_clu_center.pop(min_j)
        self.cluCenter = tmp_clu_center
        # data类要记录 相关的position
        # dis_data = bst.BstDisData(dis, i, j)  # 这里i j放的是cluCenter的角标
        # clu_min_dis_bst.try_add(dis_data)
        # dis_list = clu_min_dis_bst.postOrderTraverse(clu_min_dis_bst.root)
        # new_clu_center = create_new_clu_from_old_and_dis_list(self.cluCenter, dis_list)  # 获得新的clu center
        # self.cluCenter = new_clu_center

        pass

    def cal_min_dis_bet2candi(self, candidate_x, candidate_y):
        dis = float("inf")
        for i, x in enumerate(candidate_x):
            for j, y in enumerate(candidate_y):
                dis = min(dis, self.Dist.cal_dist(x, y, self.X[x], self.X[y]))
        return dis

    def cal_all_dis_bet2candi(self, candidate_x, candidate_y):
        dis = []
        for i, x in enumerate(candidate_x):
            for j, y in enumerate(candidate_y):
                dis.append(self.Dist.cal_dist(x, y, self.X[x], self.X[y]))
        return dis

    pass
