"""
行列を用いて連立方程式を解く
"""

import numpy as np

# 係数行列を作成
A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

# 定数ベクトルを作成
b = np.array([0, 0, 0])

# 連立方程式を解く
x = np.linalg.solve(A, b)

# 結果を表示
print("x = {}, y = {}, z = {}".format(x[0], x[1], x[2]))

