"""
重回帰モデルのモジュールを利用した多項式回帰モデル
"""

import linearreg # 編集が必要です。
import numpy as np


class PolynomialRegression:
    def __init__(self, degree):
        """
        Args:
            degree: 多項式回帰の最大次数 d
        """
        self.degree = degree
        
    def fit(self, x, y):
        # xの多項式配列を作成
        x_pow = []
        xx = x.reshape(len(x),1)
        for i in range(1, self.degree + 1):
            x_pow.append(xx**i)
        mat = np.concatenate(x_pow, axis=1) # 多項式回帰モデルを表現する特徴量行列
        linreg = linearreg.LinearRegression()
        linreg.fit(mat, y) # 重回帰モデルの特徴量行列の入力を多項式回帰の行列にする
        self.w_ = linreg.w_
        
    def predict(self, x):
        r = 0
        for i in range(self.degree + 1):
            r += x**i * self.w_[i]
        return r