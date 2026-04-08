"""
リッジ回帰モデルのモジュールを利用した多項式回帰モデル
"""

import ridge
import numpy as np


class PolyRidgeRegression:
    def __init__(self, degree, alpha=1.0):
        """
        Args:
            degree: 多項式回帰の最大次数 d
            alpha: 正則化項の係数 ハイパーパラメータ
        """
        self.degree = degree
        self.alpha = alpha
        
    def fit(self, x, y):
        # xの多項式配列を作成
        x_pow = []
        xx = x.reshape(len(x),1)
        for i in range(1, self.degree + 1):
            x_pow.append(xx**i)
        mat = np.concatenate(x_pow, axis=1) # 多項式回帰モデルを表現する特徴量行列
        ridgereg = ridge.RidgeRegression(self.alpha)
        ridgereg.fit(mat, y) # リッジ回帰モデルの特徴量行列の入力を多項式回帰の行列にする
        self.w_ = ridgereg.w_
        
    def predict(self, x):
        r = 0
        for i in range(self.degree + 1):
            r += x**i * self.w_[i]
        return r