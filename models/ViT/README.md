# ViT

## Step 1: Patch Embedding
- First steps is to convert an image into a series of patch embeddings.
- Steps:
    -  Divide the image into fixed sized patches [16x16]
    -  Flatten each patch into a vector
    -  Project each vector onto a d_model dimensional embedding using a linear layer. 

## Step 2: Positional Encodings
- Self attention treats input tokens like sets, not sequences.  There is no information contained on position or where a patch would be in the image.
- We use positional encodings to provide this information.
- There are two main types of positional encodings, but people mainly use Learnable Positional Encodings.
- nn.Parameter: Marks it as a learnable parameter that gets updated during training
- initialize a random tensor of shape [1, seq_len, d_model]
- This means each token embedding gets a position-specific bias. Essentially its a bias vector that contains info for each token.

# Step 3: Encoder Block:
-  Set up an encoder block. It is similar to a decoder block, but we are learning all information from every token, so there is no need for masking.
-  Start with a LayerNorm layer, followed by the attention, then another LN layer.
- Finally add an MLP FFN to introduce non-linearity and more complexity between our features.
- nn.Linear(d_model, mlp_dim) — projects the input vector from model dimension to a higher dimension (mlp_dim), increasing representational capacity.
- nn.GELU() — applies a non-linear activation function (Gaussian Error Linear Unit), allowing the model to learn non-linear features.
- nn.Dropout(dropout) — regularizes by randomly zeroing some activations during training, preventing overfitting.
- nn.Linear(mlp_dim, d_model) — projects back down to the original model dimension.
- Another nn.Dropout(dropout) — further regularization after projection.

# Step 4: Multi-Head Attention (Not Masked):
- We will go through the forward pass.  We assume we have already initialized our qkv weights in the init function.
- First we get out batch_size, seq_len, and embedding dim.  In this case, the embed_dim is the same as the d_k. They both represent the dimensionality of the token embeddings that pass through the multi-head attention layer.
- Compute the linear transformation of x (input) across the projected qkv layer.
- We then need to seperate the 3 vectors apart as well as divide each's tokens by the number of heads, so each head processes the same number of tokens.
- We use .view to change 3 * emb_dim  to be a vector of shape [3, n_heads, d_k].  What this does is change from 3 * emb_dim to be 3 vectors of n_head, by emb_dim // n_head. 
- We then use permute to reorder the dimenions of the tensor. we set it up to be shape [3, batch_size, n_heads, seq_len, d_k]
- We then split the q, k, and v vectors. we use these to compute the attention scores.  We get them from $q @ k^T // \sqrt{d_k}$
- This gives us the attention scores.  These are essentially logits, raw scores of how much each token attends to each other. 
- We then run it through softmax.  we only do the last dimension through softmax because each row holds an attention score for every other token in the sequence including itself. We want these logits to be probabilties to sum up to zero and also interpretable as probabilities not raw values.
- We then calculate the weighted_avg, which is taking the attention values and doing a matrix multiplication with them and the value vector.  This causes us to end up with shape [batch_size, n_heads, seq_len, d_k].
- Finally we need to recombine back to our original input size, merging the values computed by the heads back together. Goal is to get back to shape [batch_size, seq_len, emb_dim]. First we swap the head and seq_len dimentions so each token at position s has its heads grouped together. We then call .view to recombine the emb_dim representation back together. 
- Final step is we take those probabilties and project them across the output layer, to get the output dimension and number of classes.

# Step 5: Create CLS Token:
- A learnable embedding that is prepended to the sequence of patch embeddings in a ViT.
- The purpose is to aggregate global information across the entire input image.
- We initialize it as a learnable parameter, in shape [1, 1, d_model].
- Acts as a summary token to aggregate information from all patches.

# Step 6: Create ViT:
- After all of that, we create the structure for the ViT itself.
- First we pass in all the needed information into the init funciton.  We then initialize our patch embeddings layer, followed by our cls token and our position embeddings.  That allows the model to learn position specific information as well as whole image feature information.
- After that we add a dropout layer. This helps with regularization.
- We then initialize a bunch of Encoder blocks.  We initalize them equal to the number of layers that we are having.  
- Finally, with our ViT's goal of classification, we go ahead and add a final linear layer of shape [d_model, n_classes].  
- d_model is the dimensionality of the token embeddings output by the transformer layers. After processing, each token (including the special classification token) is represented as a vector of length d_model.
- The classification head takes the final embedding of the [CLS] token, which is a vector of size d_model, and maps it to a vector of size num_classes.
- This mapping is done by a linear layer (fully connected layer) with weights shaped (d_model, num_classes) so that it transforms the d_model-dimensional vector into a num_classes-dimensional vector representing class logits.

