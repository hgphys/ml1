"""
D次元入力Kクラス分類
ロジスティック回帰のクラス
"""

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, convergence_threshold=1e-4):
        self.learning_rate = learning_rate #学習率
        self.num_iterations = num_iterations #最大反復回数
        self.convergence_threshold = convergence_threshold #収束条件
        self.weights = None
    
    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def initialize_parameters(self, num_features, num_classes):
        self.weights = np.zeros((num_features, num_classes))
    
    def compute_cost(self, X, T):
        num_samples = X.shape[0]
        z = np.dot(X, self.weights)
        probabilities = self.softmax(z)
        loss = -np.sum(T * np.log(probabilities + 1e-7))
        cost = loss / num_samples
        return cost
    
    def fit(self, X, T):
        num_samples, num_features = X.shape
        num_classes = T.shape[1]
        Xtilde = np.c_[np.ones((num_samples, 1)), X]  # バイアス項を追加
        
        self.initialize_parameters(num_features + 1, num_classes)
        
        prev_cost = float('inf')
        for iteration in range(self.num_iterations):
            z = np.dot(Xtilde, self.weights)
            probabilities = self.softmax(z)
            
            dw = (1 / num_samples) * np.dot(Xtilde.T, (probabilities - T))
            self.weights -= self.learning_rate * dw
            
            cost = self.compute_cost(Xtilde, T)
            
            if np.abs(cost - prev_cost) < self.convergence_threshold:
                print(f"Converged at iteration {iteration}.")
                break
            
            prev_cost = cost
        
        return self.weights, cost
    
    def predict(self, X):
        num_samples = X.shape[0]
        Xtilde = np.c_[np.ones((num_samples, 1)), X]
        
        z = np.dot(Xtilde, self.weights)
        probabilities = self.softmax(z)
        return np.argmax(probabilities, axis=1)
