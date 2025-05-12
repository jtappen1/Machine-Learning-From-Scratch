
import numpy as np


class SVM:
    def __init__(self, lr: float, num_epochs: int = 1000 , lambda_param: float = 0.01):
        self.lr = lr
        self.num_epochs = num_epochs
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features,  = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # The regularization term: 1/2 * ||w||^2
        # The Hinge Loss: C(1-n) max(0, 1 - y_i(w * x_i + b))
        for _ in range(self.num_epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.lr * y[idx]


    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias
        return np.sign(approx)
