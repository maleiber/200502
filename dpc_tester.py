import time
from typing import Any

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


class DpcTester(object):
    def __init__(self, **kwargs):
        """
        用于测试DPC数据的类
        包括转换测试的结果以及输出测试精度
        """
        self.wrong_list = []
        if kwargs.__contains__('save_path'):
            self.save_path = kwargs['save_path']
        else:
            self.save_path = ".\\tester" + time.asctime(time.localtime(time.time()))
        if kwargs.__contains__('label'):
            self.label = kwargs['label']
        else:
            self.label = []
        if kwargs.__contains__('clu_label'):
            self.clu_label = kwargs['clu_label']
        else:
            self.clu_label = []
        try:
            pass
        except KeyError as ae:
            print('key error', ae)

    def set_label(self, label):
        self.label = label

    def gen_clu_label_from_clu_center(self, clu_center):
        # get match of result and label
        vec_list = build_n_vector_from_n_cluster(build_vector_from_DPC_cluster(clu_center), len(self.label))
        label_vec_list = build_vector_list_from_label(self.label, len(self.label))
        vec2label_match = get_clu2label_match(vec_list, label_vec_list)
        # get match of result clu and label value

        # print('match', vec2label_match)
        self.clu_label = unify_clu_and_label(vec_list, vec2label_match)
        # print('clu label', self.clu_label, 'num:', len(self.clu_label))
        return self.clu_label
        pass

    def get_score(self, d):  # d is data seq
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self.label, self.clu_label))
        print("Completeness: %0.3f" % metrics.completeness_score(self.label, self.clu_label))
        print("V-measure: %0.3f" % metrics.v_measure_score(self.label, self.clu_label))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(self.label, self.clu_label))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(self.label, self.clu_label))
        print("acc: %0.3f" % accuracy_score(self.label, self.clu_label))
        try:
            print("Silhouette Coefficient: %0.3f"
                  % metrics.silhouette_score(d, self.clu_label))
        except ValueError:
            pass

        return accuracy_score(self.label, self.clu_label)
        # self.get_wrong_label(d)

    def get_wrong_label(self, d):
        for i, x in enumerate(self.label):
            if self.clu_label[i] != x:
                self.wrong_list.append(i)
        print('wrong position:', self.wrong_list)
        print('value is:', [d[i] for i in self.wrong_list])
        return self.wrong_list

    def save_file(self):
        if len(self.label) is not 0 and len(self.clu_label) is not 0:
            pass
        pass

    pass


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
    tmp_use_dict={}
    for i in range(vec_end):
        tmp_array = []
        for j in range(len(vec_list), new_len):
            tmp_array.append([j, match_mtrx[i][j]])
            tmp_array = sorted(tmp_array, key=lambda x: x[1])
        # while tmp_use_dict.__contains__(tmp_array[-1][0]):
        #     tmp_array.pop()
        # if len(tmp_array) == vec_end - 1:
        #     match_pair.append([i, 0])
        match_pair.append([i, tmp_array[-1][0] - vec_end])
        tmp_use_dict[tmp_array[-1][0]] = 1
    return match_pair

def get_matched_label_from_pred(label, pred, label_num):
    label_vec_list = build_vector_list_from_label(label, label_num)
    vec_list = []
    for i in range(label_num):
        temp_list = []
        for y in pred:
            if y == i:
                temp_list.append(1)
            else:
                temp_list.append(0)
        vec_list.append(temp_list)
    # print(vec_list,'\n', label_vec_list)
    match = get_clu2label_match(vec_list, label_vec_list)
    # print(match_dict)
    match_dict = {}
    for i in range(max(label)+1):
        match_dict[i] = 0
    first_dict = {}
    for m in match:
        if first_dict.__contains__(m[1]):
            pass
            # continue
        match_dict[m[0]] = m[1]
        first_dict[m[1]] = 1
    # print(match,match_dict,pred)

    if max(label) == 0:  # 无监督
        return pred

    ret_label=[match_dict[i] if match_dict.__contains__(i) else 0 for i in pred]
    # ret_label=pred
    # for i in pred:
    #     try:
    #         ret_label[i] = match_dict[i]
    #     except KeyError:
    #         ret_label[i] = 0
    return ret_label

def unify_clu_and_label(vec_list, match):
    # 以label中的为准
    length = len(vec_list[0])
    new_label = [0 for x in range(length)]
    match_dict = {}
    for i, x in enumerate(match):
        match_dict[x[0]] = x[1]

    for i in range(len(vec_list)):
        now_array = vec_list[i]
        for j, y in enumerate(now_array):
            if y == 1:
                new_label[j] = match_dict[i]
    return new_label
