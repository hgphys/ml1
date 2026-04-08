"""
直線 y=ax+b によってフィットを行うプログラム
＊パラメータを解析的に決定
"""

import numpy as np
import matplotlib.pyplot as plt

# 回帰直線の定義
def reg1dim2(x, y):
    n = len(x)
    a = ((np.dot(x,y)- x.sum() * y.sum()/n)
         /((x**2).sum() - (x.sum())**2 /n))
    b = 0 # ここに正しく入力する
    return a, b

# サンプルデータ
x = np.array([1, 2, 4, 6, 7])
y = np.array([1, 3, 3, 5, 4])

# パラメータの決定
a, b = reg1dim2(x,y)

# 回帰直線付きの散布図を可視化
plt.scatter(x, y, color="b", label='sample data')
xmax = x.max()
plt.plot([0,xmax], [b, a*xmax + b], color="r", label='prediction: a={0}, b={1}'.format(a,b))

# グラフのタイトルや軸ラベルを設定
plt.xlabel('x')
plt.ylabel('y')

# 凡例を追加
plt.legend()

# グラフを表示
plt.show()