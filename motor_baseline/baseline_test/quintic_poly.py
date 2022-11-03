# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : quintic_poly.py
# Time       ：2022/11/3 9:07
# Author     ：author name
# version    ：python 3.6
# Description：
"""

# 五次多项式,需要6个约束条件求解
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.figsize']=(10,5)

matplotlib.use('TkAgg')


class quintic_poly():
    def __init__(self, p0, p1, v0, v1, a0, a1, t0, t1):
        self.p0 = p0
        self.p1 = p1
        self.v0 = v0
        self.v1 = v1
        self.a0 = a0
        self.a1 = a1
        self.t0 = t0
        self.t1 = t1

    # 求解参数
    def solve_params(self):
        T = self.t1 - self.t0
        H = self.p1 - self.p0
        a0 = self.p0
        a1 = self.v0
        a2 = 0.5 * self.a0
        a3 = 1 / (2 * (T ** 3)) * (20 * H - (8 * self.v1 + 12 * self.v0) * T - (3 * self.a0 - self.a1) * (T ** 2))
        a4 = 1 / (2 * (T ** 4)) * (-30 * H + (14 * self.v1 + 16 * self.v0) * T + (3 * self.a0 - 2 * self.a1) * (T ** 2))
        a5 = 1 / (2 * (T ** 5)) * (12 * H - 6 * (self.v1 + self.v0) * T + (self.a1 - self.a0) * (T ** 2))
        print(
            "quintic_func = {} + {} * t + {} * t ** 2 + {} * t ** 3 + {} * t ** 4 + {} * t ** 5".format(a0, a1, a2, a3,
                                                                                                        a4, a5))
        return a0, a1, a2, a3, a4, a5

    def fun1(self, ts1):
        a0, a1, a2, a3, a4, a5 = self.solve_params()
        part_0 = []
        part_1 = []
        for i in ts1:
            fun_i = a0 + a1 * i + a2 * i ** 2 + a3 * i ** 3 + a4 * i ** 4 + a5 * i ** 5
            fun_k = a1 + 2 * a2 * i + 3 * a3 * i ** 2 + 4 * a4 * i ** 3 + 5 * a5 * i ** 4
            part_0.append(fun_i)
            part_1.append(fun_k)

        return part_0, part_1
        # plt.plot(t, fun)
        # plt.plot(t,fun_der)
        # plt.show()

    def fun2(self, ts2):
        part_2 = []
        for i in ts2:
            part_2.append(30)
        return part_2

    def fun3(self, ts3):
        part_3 = []
        for i in ts3:
            part_3.append(30 + 22.5 * i)
        return part_3

    def fun4(self, ts4):
        part_4 = []
        for i in ts4:
            part_4.append(48)
        return part_4

    def fun5(self, ts5):
        part_5 = []
        for i in ts5:
            part_5.append(48 - 29 * i)
        return part_5


if __name__ == '__main__':
    ts1 = np.arange(0, 0.47, 0.01)
    ts2 = np.arange(0, 0.82, 0.01)
    ts3 = np.arange(0, 0.8, 0.01)
    ts4 = np.arange(0, 0.63, 0.01)
    ts5 = np.arange(0, 0.6, 0.01)
    ts6 = np.arange(0, 0.57, 0.01)
    p = quintic_poly(0, 30, 0, 0, 800, -800, 0, 0.47)
    p2 = quintic_poly(30, 0, 0, 0, -800, 800, 0, 0.57)
    res0 = p.fun1(ts1)[0]
    print(res0)
    res1 = p.fun1(ts1)[1]
    print(res1)
    #
    plt.plot(range(len(res0)), res0)
    plt.plot(range(len(res1)), res1)
    plt.show()

    res = p.fun1(ts1)[0]+p.fun2(ts2)+p.fun3(ts3)+p.fun4(ts4)+p.fun5(ts4)+p2.fun1(ts6)[0]
    print(p.fun1(ts1))
    plt.plot(range(len(res)),res)
    plt.xlabel('采样点')
    plt.ylabel('位置')
    plt.show()
