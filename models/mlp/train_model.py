from sklearn.datasets import fetch_openml
import numpy as np
from mlp import MLP

# Load MNIST from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

# Split into train/test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLP()
mlp.fit(X_train, y_train)

y_pred_probs = mlp.forward(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")