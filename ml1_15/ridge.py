"""
リッジ回帰モデルのクラス
"""

import numpy as np
from scipy import linalg

class RidgeRegression:
    def __init__(self, alpha = 1.0):
        """
        Args:
            w_: 回帰係数の配列 [w0, w1, ...]
            alpha: 正則化項の係数 ハイパーパラメータ
        """
        self.w_ = None
        self.alpha = alpha
        
    def fit(self, X, y):
        """
        行列表示の連立方程式Ax=bを解くことによって回帰係数を求解する
        Args:
            X: 特徴量行列等
            y: 目的変数のデータ列
        """
        Xtil = 
        identity = 
        A = 
        b = 
        self.w_ = linalg.solve(A, b) # linalg.solve(A, b)はAx=bの解を与えます
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = 
        return # Xtil と self.w_で予測を計算