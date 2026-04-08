"""
関数とその偏導関数を入力して勾配降下法によって最小値を求めるプログラム
"""

import numpy as np
import matplotlib.pyplot as plt
import gd

# 関数の定義
def f(xx):
    x = xx[0]
    y = xx[1]
    return 5 * x**2 - 6 * x * y + 3 * y**2 + 6 * x - 6 * y


# 偏導関数・勾配ベクトルの定義
def df(xx):
    x = xx[0]
    y = xx[1]
    return np.array([x + y, x + y]) # 正しい偏導関数を入力しましょう。 


# モジュールを用いて最適化の計算
algo = gd.GradientDescent(f, df, alpha = 0.2)
initial = np.array([1, 1])
algo.solve(initial)
print("(x,y)=", algo.x_)
print("f(x,y)=",algo.opt_)


"""
以下、計算過程の可視化
"""
plt.figure(figsize=(5,5))

# 最適化の計算過程を可視化
plt.scatter(initial[0], initial[1], color="k", marker="o")
plt.plot(algo.path_[:, 0], algo.path_[:, 1], color="k", linewidth=1.5)

# Contour plotの準備
x = np.linspace(-2, 2, 300)  
y = np.linspace(-2, 2, 300)  
X = np.meshgrid(x, y)
Z = f(X)

# Contour plotの作成
plt.contour(X[0], X[1], Z, levels=100)  
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel('x')
plt.ylabel('y')
plt.title('alpha = {:.2f}'.format(algo.alpha))

# グラフの表示
plt.show()