# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : diff_period.py
# Time       ：2022/11/3 9:35
# Author     ：author name
# version    ：python 3.6
# Description：
"""

import pandas as pd
import numpy as np

class get_cycle_number():
    def __init__(self, data, d):
        self.data = data
        self.d = d

    def cal_diff(self):
        y0 = data[0:-self.d];
        y1 = data[self.d:]
        diff = (y1 - y0)
        return y0, diff

    def cal_cyc_number(self,d):
        y0, diff = self.cal_diff()
        he = np.array([sum(diff[i:i + d]) for i in range(0, diff.shape[0], d)])
        ind = np.where(he == 0)[0]
        ind1 = ind * d
        ind2 = (ind + 1) * d
        diff2 = np.ones_like(diff)
        for i, j in zip(ind1, ind2):
            diff2[i:j] = 0
        ind = np.where(diff2 == 0)[0]
        y1 = np.delete(y0, ind)
        diff2 = np.delete(diff2, ind)
        lag_fftscore = Lag_FFTscore(y1)
        return lag_fftscore.result


if __name__ == '__main__':
    data = pd.DataFrame([1,2,3,0,5,6,7])
    print(np.where(data == 0)[0])