"""
gd.py
勾配降下法を用いて最小値を求めるクラス
"""

import numpy as np


class GradientDescent:
    def __init__(self, f, df, alpha=0.01, max_iterations=100, eps=1e-6):
        """
        勾配降下法を実行して最小値を求めるクラスのコンストラクタ

        Args:
            f: 目的の関数
            df: 目的の関数の導関数
            alpha: 学習率（デフォルト値: 0.01）
            max_iterations: 最大反復回数（デフォルト値: 100）
            eps: 収束判定の許容誤差（デフォルト値: 1e-6）
        """
        self.f = f
        self.df = df
        self.alpha = alpha
        self.eps = eps
        self.max_iterations = max_iterations
        self.path = None
        
    def solve(self, init):
        """
        勾配降下法を実行して最小値を求めるメソッド

        Args:
            initial: 初期値
        Notes:
            実行によって保存されるデータ属性
            x_: 最適解
            opt_: 最適解における関数値
            path_: 初期値から最適解までの経路
        """
        x = init
        path = []
        grad = self.df(x)
        max_iterations = self.max_iterations
        path.append(x)
        
        # 勾配ベクトルの大きさがeps未満になるまで更新
        for iteration in range(max_iterations):
            x = x - self.alpha * grad # 座標の更新
            grad = self.df(x) # 新しい座標で勾配を計算
            path.append(x)
            if (grad**2).sum() < self.eps**2:
                break
            
        self.path_ = np.array(path)
        self.x_ = x
        self.opt_ = self.f(x)