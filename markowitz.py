# -*- coding: utf-8 -*-

"""
This module is used to implement Markowitz model
"""

import ffn
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg


class MeanVariance:
    def __init__(self, returns):
        """定义构造器，传入收益率数据，pandas DataFrame格式"""
        self.returns = returns

    def minVar(self, goalRet):
        """定义最小化方差的函数，即求解二次规划，由于使用scipy线性方程组解法，无法加入对解的约束，
        因此使用此模型求得资产配置比存在卖空情况；
        :param goalRet: 期望收益率
        :return: 资产配置比
        """
        covs = np.array(self.returns.cov())
        means = np.array(self.returns.mean())
        L1 = np.append(np.append(covs.swapaxes(0, 1), [means], 0),
                       [np.ones(len(means))], 0).swapaxes(0, 1)
        L2 = list(np.ones(len(means)))
        L2.extend([0, 0])
        L3 = list(means)
        L3.extend([0, 0])
        L4 = np.array([L2, L3])
        L = np.append(L1, L4, 0)
        results = linalg.solve(L, np.append(np.zeros(len(means)), [1, goalRet], 0))
        return np.array([list(self.returns.columns), results[:-2]])

    def frontierCurve(self):
        """定义绘制最小方差前缘曲线函数"""
        goals = [x / 500000 for x in range(-100, 4000)]
        variances = list(map(lambda x: self.calVar(self.minVar(x)[1, :].astype(np.float)), goals))
        plt.plot(variances, goals)
        plt.show()

    def meanRet(self, fracs):
        """
        给定各资产比例，计算收益率均值
        :param fracs: 资产配置比
        :return:
        """
        meanRisky = ffn.to_returns(self.returns).mean()
        assert len(meanRisky) == len(fracs), 'Length of fractions must be equal to number of assets'
        return np.sum(np.multiply(meanRisky, np.array(fracs)))

    def calVar(self, fracs):
        """
        给定各资产比例，计算收益率方差
        :param fracs: 资产配置比
        :return:
        """
        return np.dot(np.dot(fracs, self.returns.cov()), fracs)
