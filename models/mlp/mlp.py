
import numpy as np


class MLP:
    def __init__(self, lr = 0.001, batch_size = 64, num_epochs=25):
        self._lr = lr
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        # Initialize the weights
        self._weights_1 = None
        self._bias_1 = None
        self._weights_2 = None
        self._bias_2 = None

    def relu(self, x):
        # Implementation of the ReLU activation function
        return np.maximum(0, x)
    
    def softmax(self, x):
        # Implementation of the softmax activation function
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy(self, y_pred, y_true):
        # Implementation of Cross-Entropy Loss
        # y-true shape 1, 64
        n_samples = y_pred.shape[0]                         # batch_size, num_classes
        logp = -np.log(y_pred[range(n_samples), y_true])    # Log takes the numbers from 0 to 1, and stretches it from 0 to -infinity
        loss = np.sum(logp) / n_samples                     # Computes a single value for a entire batch
        return loss
    
    def forward(self, X):
        """
        Forward pass through the model.  Computes the linear transformations and non-linear activations of the layers.
        """
        # Shape of X is (batch_size, input_size)
        # All of the Zs are Linear Transformation
        # All of the activations are Non-Linear Actications
        z_1 = X @ self._weights_1 + self._bias_1
        a_1 = self.relu(z_1)

        z_2= a_1 @ self._weights_2 + self._bias_2
        a_2 = self.relu(z_2)

        z_3 = a_2 @ self._weights_out + self._bias_out
        a_3 = self.softmax(z_3)

        return z_1, a_1, z_2, a_2, z_3, a_3

    def backwards(self, X, y, z1, a1, z2, a2, z3, out):
        """
        Back-propagation using the chain-rule back through the model.
        """
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
        """
        Fit function runs the data through the model.
        """
        num_samples, num_features = np.shape(X)     # [_, 784]
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
            # Shuffle the data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            epoch_loss = 0
            num_batches = num_samples // self._batch_size
            # Seperate the data into batches and run the forward and backward pass per batch
            for i in range(0, num_samples, self._batch_size):
                X_batch = X[i:i + self._batch_size]
                y_batch = y[i:i + self._batch_size]     # shape 1, batch_size

                # Do the forward pass and backpropagation
                z1, a1, z2, a2, z3, out = self.forward(X_batch) 
                loss = self.cross_entropy(out, y_batch)     # out: [batch_size, num_classes]
                self.backwards(X_batch, y_batch, z1, a1, z2, a2, z3, out)

                epoch_loss += loss

            # Calculate the average loss to get a representative sample of how the model did
            avg_loss = epoch_loss / num_batches
            print(f"Finished Epoch {epoch + 1} with average loss: {avg_loss:.4f}")



        