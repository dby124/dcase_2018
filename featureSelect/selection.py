import numpy as np
import math as mt
import random as rd
# from .evaluation import evaluation


EPS = 2.2204e-16

# 群智能特征选择算法参数设置
FROG_NUM = 15
GROUP_NUM = 3
FROG_IN_GROUP = (FROG_NUM//GROUP_NUM)
REPEAT_NUM = 1000
FEATURE_NUM = 10		# 特征初始集的个数
FEATURE_SUB_NUM = 5	    # 特征子集的个数
DATA_NUM = 10



def ami(x, y):
    """
    Calculates the mutual average information of x ang y with a possible lag.
    :param x: the time series.
    :param y: the time series.
    :return: the normalized average mutual information, it means how many bit x and y has in common relative to
             how many bits is needed for the internally binned representation of x or y.
    """
    # 验证输入参数的有效性
    # print("输出x，y的长度：", len(x), len(y))
    try:
        (len(x) == len(y))
    except Exception as e:
        print("输入的两个变量的长度不一致！")

    summ = 0
    data_num = len(x)

    # 对数据进行归一化
    x = (x - min(x)) * (1 - EPS) / max(x - min(x))
    y = (y - min(y)) * (1 - EPS) / max(y - min(y))

    # 初始化binx，biny
    bins = mt.floor(1 + mt.log(data_num, 2) + 0.5)

    binx = [mt.floor(x[0] * bins)]
    biny = [mt.floor(y[0] * bins)]

    for i in range(data_num):
        if i > 0:
            binx.append(mt.floor(x[i] * bins))
            biny.append(mt.floor(y[i] * bins))

        if (binx[i] > bins) or (binx[i] == bins):
            binx[i] = bins - 1

        if (biny[i] > bins) or (biny[i] == bins):
            biny[i] = bins - 1

    # 求联合概率pxy
    pxy = np.zeros((bins, bins), float)
    px = np.zeros(bins, float)
    py = np.zeros(bins, float)

    for i in range(data_num):
        pxy[binx[i]][biny[i]] = (pxy[binx[i]][biny[i]] + 1)
        # print(pxy[binx[i]][biny[i]])

    for i in range(len(pxy[0])):
        for j in range(len(pxy[0])):
            pxy[i][j] = (pxy[i][j] / data_num) + EPS

    # 求px，py
    for i in range(len(pxy[0])):
        for j in range(len(pxy[0])):
            px[i] = px[i] + pxy[i][j]   # 对行求和
            py[i] = py[i] + pxy[j][i]   # 对列求和

    # 求x与y的平均互信息量
    for i in range(len(pxy[0])):
        for j in range(len(pxy[0])):
            summ = summ + (pxy[i][j] * (mt.log((pxy[i][j] / (px[i] * py[j] + EPS)), 2)))

    return summ / mt.log(bins, 2)


def evaluation(data, label, B):
    """
    calculate evaluation value of feature subset.
    :param data: numpy array, sample data, the rows represent the samples and the columns represent features.
    :param label:numpy array, the class label of samples.
    :param B: the scheme of feature subset.
    :return: eva, the evaluation value of feature subset.
    """
    data = np.array(data)
    label = np.array(label)
    #
    x = data[:, B]
    [m, n] = x.shape     # 此处m为样本个数，n对应特征的个数
    fmi_num = sum(range(1, n))

    sum_lmi = 0
    sum_fmi = 0

    # 求互信息量
    for i in range(n):
        sum_lmi = sum_lmi + ami(x[:, i], label)
        for j in range(n):
            if i > j:
                sum_fmi = sum_fmi + ami(x[:, i], x[:, j])
            else:
                break

    ave_mi = sum_fmi / fmi_num
    # eva = sum_lmi - ave_mi
    # print(eva)
    return sum_lmi - ave_mi


class Frog(object):
    def __init__(self):
        self.pos = np.array(rd.sample(range(FEATURE_NUM), FEATURE_SUB_NUM))  # 随机从初始特征集中随机选取一个特征子集
        self.eva = evaluation(data, label, self.pos)

    # @property
    def set(self, arr):
        self.pos = arr
        self.eva = evaluation(data, label, self.pos)
        # print("更新后:", self.pos, self.eva)

    def show(self):
        print("show:", self.pos, self.eva)


def readtxt(filename, row, col):
    # filename = 'test_data.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径
    f = open(filename, 'r')
    result = list()
    for line in open(filename):
        for i in line.split():
            result.append(i)

    data = np.zeros((row, col), float)
    k = 0
    for i in range(row):
        for j in range(col):
            data[i][j] = result[k]
            k = k + 1

    f.close()
    return data


def init_frogs(length):
    arr = []
    for i in range(length):
        arr.append(Frog())

    return arr


# 加载数据
data = readtxt("test_data.txt", DATA_NUM, FEATURE_NUM)
label = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# init frog group
global_max = Frog()                     # the best in global
global_max.show()
max_in_group = init_frogs(GROUP_NUM)    # the worst
min_in_group = init_frogs(GROUP_NUM)    # the best
allfrog = init_frogs(FROG_NUM)
grouped = np.zeros((GROUP_NUM, FROG_IN_GROUP), Frog)    # grouped group


def sort():
    global global_max
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
    # global_max.show()
    for i in range(GROUP_NUM):
        max_in_group[i] = grouped[i][0]
        min_in_group[i] = grouped[i][FROG_IN_GROUP - 1]


def update():
    for i in range(GROUP_NUM):
        temp = min_in_group[i]
        # print("origin")
        # temp.show()
        new_pos = updated_pos(temp.pos, max_in_group[i].pos)    # 往组内最优方向跳
        # print("first")
        temp.set(new_pos)
        print(temp.pos, temp.eva, new_pos, min_in_group[i].eva)
        if temp.eva > min_in_group[i].eva:  # 跳跃成功
            grouped[i][FROG_IN_GROUP - 1] = temp
        else:
            temp = min_in_group[i]
            new_pos = updated_pos(temp.pos, global_max.pos)  # 往全局最优方向跳
            # print("second")
            temp.set(new_pos)
            print(temp.pos, temp.eva, new_pos, min_in_group[i].eva)
            if temp.eva > min_in_group[i].eva:  # 跳跃成功
                grouped[i][FROG_IN_GROUP - 1] = temp
            else:
                # print("third")
                temp.set(rd.sample(range(FEATURE_NUM), FEATURE_SUB_NUM))
                print(temp.pos, temp.eva, new_pos, min_in_group[i].eva)
                grouped[i][FROG_IN_GROUP - 1] = temp


def updated_pos(pos, dir_pos):
    same = list(set(pos).intersection(set(dir_pos)))
    diff1 = list(set(pos).difference(set(dir_pos)))
    diff2 = list(set(dir_pos).difference(set(pos)))
    new_pos = np.array(same + diff1)
    for index, val in enumerate(diff2):
        if rd.random() > 0.5:
            new_pos[len(same)+index] = val

    print(same+diff1, same+diff2, new_pos)
    return new_pos


def shuffle():
    k = 0
    for j in range(FROG_IN_GROUP):
        for i in range(GROUP_NUM):
            allfrog[k] = grouped[i][j]
            k = k + 1


def select():
    sort()
    for ite in range(REPEAT_NUM):
        update()
        shuffle()
        sort()
        print(ite, global_max.pos, global_max.eva)

    return global_max.pos, global_max.eva


# 群智能选择过程
select()
print(global_max.pos, global_max.eva)

print(evaluation(data, label, np.array([7, 1, 6, 9, 8])))
