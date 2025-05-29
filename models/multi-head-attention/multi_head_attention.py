
# Example input data.  Each Token is a 8 dimensional vector. Shape: [3, 8]
import numpy as np

X = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 2, 0, 2, 0, 2, 0, 2],
    [1, 1, 1, 1, 1, 1, 1, 1]
])
# Add a batch size dimension. Shape: [1,3,8]
X.reshape(1, 3, 8)

batch_size = 1  # The bumber of data values there are in our input
num_heads = 4   # The number of heads
seq_len = 3     # The length of the sequence aka how many tokens there are in the input vector
d_model = 8     # The number of features that represent a token
d_k = d_model // num_heads      # The dimension of the K vector

# Initialize the Q,K,V weights
W_q = np.random.rand(d_model, d_model)  # Shape: [8,8]
W_k = np.random.rand(d_model, d_model)  # Shape: [8,8]
W_v = np.random.rand(d_model, d_model)  # Shape: [8,8]

# Calculate Q, K, V
Q = X @ W_q      # Shape: [1,3,8]
K =  X @ W_k     # Shape: [1,3,8]
V = X @ W_v      # Shape: [1,3,8]

# Reshape the weights to work between heads.
# Reshape to  [batch_size, num_heads, sequence_length, d_head], splitting the token into 4 groups of 2 features.
Q_split = Q.reshape(batch_size, num_heads, seq_len, d_k)       # Shape: (1, 3, 4, 2)
# Transpose so each head gets all 3 tokens. shape: (batch_size, num_heads, seq_len, d_k)
Q_heads = Q_split.transpose(0, 2, 1, 3)   # Shape: (1, 4, 3, 2)

# Do the same for K and V
K_split = K.reshape(batch_size, num_heads, seq_len, d_k)
K_heads = K_split.transpose(0, 2, 1, 3)

V_split = V.reshape(batch_size, num_heads, seq_len, d_k)
V_heads = V_split.transpose(0, 2, 1, 3)

def compute_scaled_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 1, 3, 2)  # (batch, heads, seq, seq)
    scores = scores / np.sqrt(d_k)
    
    # Softmax along last axis (seq_len)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    output = weights @ V  # (batch, heads, seq_len, d_k)
    return output, weights

attention_output, attention_weights = compute_scaled_attention(Q_heads, K_heads, V_heads)

def combine_heads(x):
    batch_size, num_heads, seq_len, d_k = x.shape
    x = x.transpose(0, 2, 1, 3)  # → (batch_size, seq_len, num_heads, d_k)
    x = x.reshape(batch_size, seq_len, num_heads * d_k)
    return x

multi_head_output = combine_heads(attention_output)

W_o = np.random.rand(d_model, d_model)
final_output = multi_head_output @ W_o  # Shape: (1, 3, 8)





