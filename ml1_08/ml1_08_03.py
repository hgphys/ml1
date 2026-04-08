"""
テストデータによる検証、決定係数と残差プロット
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# model_params.csv の読み込みと回帰直線の定義
model_params = pd.read_csv('ここにファイルパスを入力する')
w0 = model_params['w0'][0]
w1 = model_params['w1'][0]

# training_data.csv の読み込みと予測値、残差の計算
training_data = pd.read_csv('data/training_data.csv')
training_data['prediction'] = w0 + w1 * training_data['x']
training_data['residual'] = training_data['y'] - training_data['prediction']
training_data.to_csv('prediction_training.csv', index=False)

# 訓練データの残差の分散
print("Var[e_training] = ", round(training_data['residual'].var(),2))

# 訓練データを用いた決定係数の計算
r_squared = training_data['prediction'].var()/training_data['y'].var()
print("R^2 = ", round(r_squared,3))

# test_data.csv の読み込みと予測値、残差の計算
test_data = pd.read_csv('data/test_data.csv')
test_data['prediction'] = w0 + w1 * test_data['x']
test_data['residual'] = test_data['y'] - test_data['prediction']
test_data.to_csv('prediction_test.csv', index=False)

# テストデータの残差の分散
print("Var[e_test] = ", round(test_data['residual'].var(),2))

# 残差プロットの可視化
plt.figure(figsize=(6, 6))
plt.scatter(training_data['prediction'], training_data['residual'], label='Training Data')
plt.scatter(test_data['prediction'], test_data['residual'], label='Test Data')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Prediction')
plt.ylabel('Residual')
plt.legend()
plt.title('Residual Plot')
plt.grid(True)
plt.show()
