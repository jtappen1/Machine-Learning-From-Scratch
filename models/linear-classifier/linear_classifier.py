import numpy as np

class LinearClassifier:
    def __init__(self, lr: float = 0.005, num_classes: int = 10, num_epochs: int = 100):
        self._lr = lr
        self._num_classes = num_classes
        self._num_epochs = num_epochs
        
        self._w_1 = None
        self._b_1 = None

    def softmax(self, z):
        # axis  = 1 means that we are summing across columns
        # keepdims is necessary for broadcasting, keeps the shape
        # subtract max to stop overflow
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z/np.sum(exp_z, axis=1, keepdims=True)


    def cross_entropy_loss(self, y, y_hat):
        # First we get the number of samples
        n_samples = y_hat.shape[0]
        # We next compute the log, for every class in y_hat 
        # we get the predicted probability of the correct class, for each sample.
        logp = -np.log(y_hat[range(n_samples), y])
        return np.sum(logp) / n_samples
        
    
    def forward(self, X):
        # linear transform through the 1 linear layer
        z1 = X @ self._w_1 + self._b_1
        # Apply softmax to the transformation.
        a1 = self.softmax(z1)
        return z1, a1
    
    def backwards(self, X, y, z1, a1):
        num_samples =  X.shape[0]
        y = np.zeros_like(a1)
        y = [np.arange(num_samples), y] = 1

        dz = (a1 - y) / num_samples
        dw = X.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)

        self._w_1 -= self._lr * dw
        self._b_1 -= self._lr * db
        

    def fit(self, X, y):
        _, num_features = X.shape
        linear_layer = 10

        self._w_1 =  np.random.randn(num_features, linear_layer) * np.sqrt(2 / num_features)
        self._bias_1 = np.zeros((linear_layer,))

        for _ in range(self._num_epochs):
            z1, a1 = self.forward(X=X)
            loss = self.cross_entropy_loss(X, y)
            self.backwards(X, y, z1, a1)

            print(f"Epoch {_ + 1}: Loss {loss:.4f}")
            
        

        


