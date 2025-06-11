# Transformer
## Decoder Steps:
- Create Multi-Head Attention
- Create and add LayerNorm Layer
- Add first Residual Connection
- Create Feed-Forward Network (MLP)
- Add second LayerNorm Layer.
- Add second Residual Connection

## Multi-Head Attention
Inputs:
- d_model: Embedding Dimension.  Aka how many features in each token's vector space. 
- n_heads: Number of attention heads that the model has.
- seq_len: The number of tokens that are processed at once.

In our Init function, we set up:
- d_k: d_model // n_heads (This tells us the dimension of the K (and Q) vectors that we will be computing attention with)
- Q, K, V linear Layer: Instead of using 3 seperate layers, we use a single layer of shape [d_model, d_model * 3] to project the Q,K, and V layers beside each other. 
- Output layer:  Also set up the output layer of shape [d_model, d_model].  This is our output dimension.

Linear Layers:
- Torch Linear layers are fully connected dense layers.  
- They apply the linear transformation $output=xW^T+b$
- so our weight matrix is of shape [1536, 512], and our bias is of shape (1536,_)

Forward function:
- X is our input, which is of shape (batch_size, sequence_length, embed_dim)
- We take our X input vector and project each token across the weights.
- This is the same as taking the $Q @ W_q^T$ but for k and V as well.
- We end up with qkv as shape (batch_size, seq_len, 3 * embdim).  So essentially for every token there are now features that contain information for Q, K, and V.
- we then use .view to reshape the tensor without changing any of it's data.
- It is being reshaped to be split evenly between the heads. It is going to shape [B, S, 3(one for each Q,K, V vector), n_heads(splitting up evenly between heads), d_k (we need to keep the dimensions the same so if heads is 8 and emb_dim = 512, our d_k would be 512/8 so in the end it would be equal to (1,512))]
- We then change the dimension order. From: (B, seq_len, 3, n_heads, head_dim) To: (3, B, n_heads, seq_len, head_dim)
- We do this and then split into individual Q, K, and V vectors. They are of shape (B, n_heads, seq_len, head_dim)
- Now we follow the other steps of attention.  We do Q @ K^T to get us the raw attention scores. How much each token should attend to each other token.
- You end up with a [seq_len, seq_len] matrix.  The matrix represents how much each token attends to every other token.  This is also per head, and per sample in the batch.
- The final matrix tells you, for each token in the sequence, how much attention it pays to every other token (including itself).
- we divide by $\sqrt{d_k}$ to prevent the dot products from getting too large.  When we divide by it, it is applied to every singe element in the matrix.
- We then take those raw attention scores, and put it through softmax in the final dimension.  What this does and for every token, it takes the softmax of it to turn them essentially into a type of normalized probabilites.
- Finally we take our attention weights and do the dot product with V.  V is the value vectors, the representations that actually get mixed together based on how similar their keys are to the query.  This allows us to get probabilities that are weighted, with the softmax telling us how much attention to pay to other tokens and the values containing semantic information on every other token.
- After this we have out.shape = (B, n_heads, S, d_k). We transpose 1, and 2 to get shape (B, S, n_heads, d_k). Then use .view() to get to shape (B, S, E).
- Finally we do a linear transformation across the last dimension of out. out = out @ W.T + b. This is done for each token in the batch.

## LayerNorm
- Stands for Layer Normalization: A technique that stablizes and speeds up training by normalizing the features of a single token.  This helps to prevent large variations in activation magnitudes which can destablize training.

How to do it:
- Compute mean and variance (Average squared deviation from the mean) over all of the features in the vector.

## Residual Connections
- Also called Skip Connections, these connections are used to help neural networks train better and faster.
- Instead of just passing data through a layer like $output = Layer(x)$, a residual connection adds the original input back to the output $output = $x + Layer(x)$


## Creating the Decoder block
- First create the MLP. We are simply using a simple 2 layer network, linear -> Relu -> linear
- The MLP adds nonlinearity and depth to the model.  