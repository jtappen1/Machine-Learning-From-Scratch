

import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Input  shape 3, 4
x = np.array([
    [1.0, 0.0, 1.0, 0.0],  # token 1
    [0.0, 2.0, 0.0, 2.0],  # token 2
    [1.0, 1.0, 1.0, 1.0]   # token 3
])

# initialize our weight matricies
W_q = np.array([[0.1, 0.0], [0.0, 0.1], [0.1, 0.0], [0.0, 0.1]])    # shape (4,2)
W_k = np.array([[0.1, 0.0], [0.0, 0.1], [0.1, 0.0], [0.0, 0.1]])
W_v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

# We next compute Q, K, V shape 3, 2
Q = x @ W_q
K = x @ W_k
V =  x @ W_v

d_k = K.shape[1]
attention_scores = Q @ K.T  # shape 3 x 3

scaled_scores = attention_scores / np.sqrt(d_k)

attention_weights = softmax(scaled_scores)

output = attention_weights @ V

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)
print("Attention Weights:\n", attention_weights)
print("Output:\n", output)