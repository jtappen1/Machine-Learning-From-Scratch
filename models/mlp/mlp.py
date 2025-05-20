
import numpy as np


class MLP:
    def __init__(self, lr = 0.001, batch_size = 64, num_epochs=100):
        self._lr = lr
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        self._weights_1 = None
        self._bias_1 = None

        self._weights_2 = None
        self._bias_2 = None

    def relu(self, x):
        # Takes each element in the list x and takes the max between it and 0.
        return np.maximum(0, x)
    
    def softmax(self, x):
        # TODO: Go back and understand this
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy(self, y_pred, y_true):
        n_samples = y_pred.shape[0]
        logp = -np.log(y_pred[range(n_samples), y_true])
        loss = np.sum(logp) / n_samples
        return loss
    
    def forward(self, X):
        # Shape of X is (batch_size, input_size)
        # Forward pass through the first hidden layer
        # All of the Zs are linear transformations
        z1 = X @ self._weights_1 + self._bias_1
        activation_1 = self.relu(z1)

        z2= activation_1 @ self._weights_2 + self._bias_2
        activation_2 = self.relu(z2)

        z3 = activation_2 @ self._weights_out + self._bias_out
        activation_3 = self.softmax(z3)

         
        return z1, activation_1, z2, activation_2, z3, activation_3

    def backwards(self, X, y, z1, a1, z2, a2, z3, out):
        # we compute backprop by doing the chain rule back through to compute the gradients
        # Chain rule y = f(u),  u = g(x)  == (y = f(g(x)))
        # dy/dx = dy/du * du/dx
        num_samples = X.shape[0]
        y_true = np.zeros_like(out)
        # This is a one hot encoding of the labels
        y_true[np.arange(num_samples), y] = 1

        # In backprop, you start at the computing the gradient
        # The gradient of Cross Entropy loss with softmax  =  (1/n)(softmax- y_one_hot)
        # This is the gradient of the loss with respect to logits
        # The goal is to update weights through the gradient of the loss.  The loss is the last thing we compute, so we start with it. 
        # W1 was updated in the first transformation, so we have to go through the intermediate steps of the chain rule.
        dz3 = (out - y_true) / num_samples
        dw_out = a2.T @ dz3                 # shape: (hidden_dim2, num_classes)
        db_out = np.sum(dz3, axis=0)        # shape: (num_classes,)

        da2 = dz3 @ self._weights_out.T
        dz2 = da2 * (z2 > 0)

        dw2 = a1.T @ dz2                     # shape: (hidden_dim1, hidden_dim2)
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self._weights_2.T
        dz1 = da1 * (z1 > 0)

        dw1 = X.T @ dz1                     # shape: (input_dim, hidden_dim1)
        db1 = np.sum(dz1, axis=0)

        self._weights_out -= self._lr * dw_out
        self._bias_out -= self._lr * db_out

        self._weights_2 -= self._lr * dw2
        self._bias_2 -= self._lr * db2

        self._weights_1 -= self._lr * dw1
        self._bias_1 -= self._lr * db1


    def fit(self, X, y):    
        num_samples, num_features = np.shape(X)
        hidden_layer_1 = 256
        hidden_layer_2 = 128
        output_size = 10

        # Input [1, 784] -> Hidden Layer [1, 256]
        # Use He / Kaiming Initialization
        self._weights_1 = np.random.randn(num_features, hidden_layer_1) * np.sqrt(2 / num_features)
        self._bias_1 = np.zeros((hidden_layer_1,))

        #  Hidden Layer 1 [1, 256] -> Hidden Layer 2 [1, 128]
        self._weights_2 = np.random.randn(hidden_layer_1, hidden_layer_2) * np.sqrt(2 / hidden_layer_1)
        self._bias_2 = np.zeros((hidden_layer_2,))

        # Hidden Layer 2 [1, 128] -> Output [1, 10]
        self._weights_out = np.random.randn(hidden_layer_2, output_size) * np.sqrt(2 / hidden_layer_2)
        self._bias_out = np.zeros((output_size,))

        for epoch in range(self._num_epochs):
            # We first compute the 
            z1, a1, z2, a2, z3, out = self.forward(X)
            loss = self.cross_entropy(out, y)
            self.backwards(X, y, z1, a1, z2, a2, z3, out)

            print(f"Finished Epoch: {epoch} with loss of {loss:.4f}")



        