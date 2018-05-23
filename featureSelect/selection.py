import numpy as np
import random as rd
from .evaluation import evaluation


# 群智能特征选择算法参数设置
FROG_NUM = 15
GROUP_NUM = 3
FROG_IN_GROUP = (FROG_NUM//GROUP_NUM)
REPEAT_NUM = 1000
FEATURE_NUM = 50		# 特征初始集的个数
FEATURE_SUB_NUM = 5	    # 特征子集的个数


class Frog(object):
    # def __init__(self, pos):
    #     self.pos = pos
    #     self.eva = evaluation(data, label, pos)
    def __init__(self):
        self.pos = rd.sample(range(FEATURE_NUM), FEATURE_SUB_NUM)  # 随机从初始特征集中随机选取一个特征子集
        self.eva = evaluation(data, label, self.pos)

    # @property
    def set(self, arr):
        self.pos = arr
        self.eva = evaluation(data, label, self.pos)


def init_frogs(length):
    arr = []
    for i in range(length):
        allfrog.append(Frog())

    return arr


# init frog group
global_max = Frog()                     # the best in global
max_in_group = init_frogs(GROUP_NUM)	# the worst
min_in_group = init_frogs(GROUP_NUM)	# the best
allfrog = init_frogs(FROG_NUM)
grouped = np.zeros((GROUP_NUM, FROG_IN_GROUP), Frog)	# grouped group


def sort():
    # 降序排列所有青蛙的eva
    for i in range(FROG_NUM):
        for j in range(FROG_NUM):
            if allfrog[i].eva < allfrog[j].eva:
                temp = allfrog[i]
                allfrog[i] = allfrog[j]
                allfrog[j] = temp

    k = 0
    # 进行分组
    for j in range(FROG_IN_GROUP):
        for i in range(GROUP_NUM):
            grouped[i][j] = allfrog[k]
            k = k + 1

    global_max = allfrog[0]
    for i in range(GROUP_NUM):
        max_in_group[i] = grouped[i][0]
        min_in_group[i] = grouped[i][FROG_IN_GROUP - 1]


def update():
    for i in range(GROUP_NUM):
        temp = min_in_group[i]
        new_pos = updated_pos(temp.pos, max_in_group[i].pos)    # 往组内最优方向跳
        temp.set(new_pos)
        if temp.eva > min_in_group[i].eva:  # 跳跃成功
            grouped[i][FROG_IN_GROUP - 1] = temp
        else:
            temp = min_in_group[i]
            new_pos = updated_pos(temp.pos, global_max.pos)  # 往全局最优方向跳
            temp.set(new_pos)
            if temp.eva > min_in_group[i].eva:  # 跳跃成功
                grouped[i][FROG_IN_GROUP - 1] = temp
            else:
                grouped[i][FROG_IN_GROUP - 1].set(rd.sample(range(FEATURE_NUM), FEATURE_SUB_NUM))


def updated_pos(pos, dir_pos):
    same = set(pos).intersection(set(dir_pos))
    diff1 = set(pos).difference(set(dir_pos))
    diff2 = set(dir_pos).difference(set(pos))
    new_pos = [same, diff1]
    for index, val in diff2:
        if rd.random() > 0.5:
            new_pos[len(same)+index] = val

    return new_pos


def select():
    sort()
    for ite in range(REPEAT_NUM):
        update()
        sort()

    return global_max.pos, global_max.eva
