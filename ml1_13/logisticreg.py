"""
ロジスティック回帰のクラス
ニュートン法の利用
"""

import numpy as np
from scipy import linalg

THRESHMIN = 1e-10

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, tol=0.001, max_iter=3):
        self.tol = tol
        self.max_iter = max_iter
        self.w_ = None
        
    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1] + 1) # 重みの初期値は全て0
        Xtil = np.c_[np.ones(X.shape[0]), X]
        diff = np.inf
        w_prev = self.w_
        iter = 0
        while diff > self.tol and iter < self.max_iter:
            yhat = sigmoid(np.dot(Xtil, self.w_))
            # ニュートン法のためのヘッセ行列の定義
            r = np.clip(yhat * (1 - yhat), THRESHMIN, np.inf) #対角行列、閾値の利用
            XR = Xtil.T * r
            XRX = np.dot(XR, Xtil)
            w_prev = self.w_
            b = np.dot(XR, np.dot(Xtil, self.w_) - 1/r * (yhat - y))
            self.w_ = linalg.solve(XRX, b) #重みの更新
            diff = abs(w_prev - self.w_).mean()
            iter += 1
        
    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        yhat = sigmoid(np.dot(Xtil, self.w_))
        return np.where(yhat > .5, 1, 0)