import numpy as np
import math as mt


EPS = 2.2204e-16
# DATA_NUM = 800


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

