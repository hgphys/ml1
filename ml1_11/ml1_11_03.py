"""
２層パーセプトロンの入力と出力
"""

import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

class TwoLayerPerceptron:
    def __init__(self, weights1, weights2, theta):
        self.weights1 = weights1
        self.weights2 = weights2
        self.theta = theta

    def activate(self, x):
        hidden_layer_input =  self.weights1 @ x - self.theta 
        hidden_layer_output = step_function(hidden_layer_input)
        output_layer_input =  self.weights2 @ hidden_layer_output - self.theta
        output_layer_output = step_function(output_layer_input)
        return output_layer_output

# パーセプトロンの重みと閾値を設定 (修正が必要)
weights1 = np.array([[1, 1], [1, 1]])
weights2 = np.array([[1, 1]])
theta = np.array([0.5])

# ２層パーセプトロンを作成
perceptron = TwoLayerPerceptron(weights1, weights2, theta)

# 入力データ
input_data = np.array([[0, 0],[1, 0],[0, 1],[1, 1]])

for i in range(len(input_data)):
    # パーセプトロンの出力を計算
    output = perceptron.activate(input_data[i])
    print(f"入力データ: {input_data[i]}")
    print(f"出力結果: {output[0]}")
