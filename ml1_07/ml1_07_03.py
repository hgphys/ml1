"""
二項分布の尤度関数に対する最尤推定
"""

import numpy as np
import scipy.optimize as opt

def likelihood(theta, n, k):
    # 二項分布の尤度関数を定義する
    p = theta
    likelihood_value = 0.5 # ここに正しく尤度関数を定義してください。
    return likelihood_value

def negative_log_likelihood(theta, n, k):
    # 負の対数尤度関数を定義する（最小化問題として扱うため）
    return -np.log(likelihood(theta, n, k))

n = 600  # 射数
k = 48  # 10点を取った射数

# 最尤推定を行い、Xを求める
result = opt.minimize_scalar(negative_log_likelihood, bounds=(0, 1), args=(n, k), method='bounded')
theta_hat = result.x

print("最尤推定された確率:")
print("theta =", theta_hat)
