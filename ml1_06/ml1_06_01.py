"""
定義した関数の数値微分を行うプログラム
"""

import numpy as np

# 中心差分で定義された数値微分による勾配ベクトルの計算
def numerical_gradient(f, x):
    h = 1e-50
    grad = np.zeros_like(x) # xと同じ形状の配列を生成
    
    #　各成分毎に差分を計算する
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h) の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2)/ (2*h)
        x[idx] = tmp_val # 値を元に戻す
        
    return grad

# 関数の定義
def f(x):
    return x[0]**2 + x[1]**2

# 結果の出力
x0 = np.array([3, 4])
print(x0, "の勾配")
print(numerical_gradient(f, x0))
