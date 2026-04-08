"""
単純パーセプトロンの入力と出力
"""

import numpy as np

class SimplePerceptron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, x):
        return 1 if np.dot(self.weights, x) >= self.threshold else 0

# パラメータを設定 (修正が必要)
weights = np.array([0.5, 0.5])
threshold = 0.5

# 単純パーセプトロンを作成
perceptron = SimplePerceptron(weights, threshold)

# 入力データ
input_data = np.array([[0, 0],[1, 0],[0, 1],[1, 1]])

for i in range(len(input_data)):
    # パーセプトロンの出力を計算
    output = perceptron.activate(input_data[i])
    print(f"入力データ: {input_data[i]}")
    print(f"出力結果: {output}")
