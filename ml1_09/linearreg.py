"""
重回帰モデルのクラス
"""

import numpy as np
from scipy import linalg

class LinearRegression:
    def __init__(self):
        """
        Args:
            w_: 回帰係数の配列 [w0, w1, ...]
        """
        self.w_ = None
        
    def fit(self, X, y):
        """
        行列表示の連立方程式Ax=bを解くことによって回帰係数を求解する
        Args:
            X: 特徴量行列
            y: 目的変数のデータ列
        """
        Xtil = np.c_[np.ones(X.shape[0]),X]
        A = # 正しく定義しましょう
        b = # 正しく定義しましょう
        self.w_ = linalg.solve(A, b) # linalg.solve(A, b)はAx=bの解を与えます
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]),X]
        return np.dot(Xtil, self.w_)