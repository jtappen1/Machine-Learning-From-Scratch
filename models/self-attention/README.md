# Self-Attention Tutorial

Starts with a sequence of text embeddings (word vectors)

Shape is generally
[batch_size, sequence_length, embedding_dim]

Input:  An sequence of tokens
Queries (Q) What the token is looking for
Keys (K) What the token has to offer
Values (V) The information the token has

Score is computed by Score = np.dot(Q, K.T)

To avoid large values, scale the score: scaled =  np.dot(Q, K.T) / np.sqrt(d_k).
We do this because Softmax applifies large values, so this is a sort of pre-normalization before softmax.