# Multi-Layer Perceptron From Scratch
## Goal:
- Understand the theory and logic behind a MLP, and develop one from scratch using Numpy.

### Notes:
- Input Layer - > Hidden Layer(s) -> Output Layer
- Linear transformation, followed by a non-linear activation
- The linear transformation is z= X*W + b (feed forward)
- When building these models, in the init the weights and biases for every hidden layer need to be initalized.
- ReLU: For 
- A one-hot encoded matrix is a way to represent a list of categorical labels (like class numbers) using binary vectors. Each row in the matrix corresponds to one sample, and each column corresponds to a class. The row contains all 0s except for a 1 in the position of the correct class.

## Activation Functions:
- Softmax:  Best for multi-class classification
- ReLU: Introduces Non-linearity


## Creating Models from Scratch:
- When initalizing the weights, you never want to initialize it as a vector of zeros. 
- Use He or Kaiming Initialization with ReLU activations: np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
- The weights and bias are of [1, num_features_in_layer].
- The Model has shape Input [784] - > Hidden [256] -> Hidden [128] - > Output [10]
- The input vector X has shape (batch_size, input_size)
- 1. Determine the different sizes and initialize 


## Notes
- You use the backpropagation to compute the gradients and gradient descent to optimize the loss function to find the weights.
- Pointwise means it is applied to every element in the matrix
- Relu means for every element in the matrix, you take the max between 0 and the element
- At the end of my forward pass, we have computed the prediction of the model
- For Softmax: 
    - All values are between 0 and 1
    - All values equal up to 1
    - Larger values are amplified
    - Ends up being a probability vector of the different classes

- Softmax is at the end of the MLP because it converts the logits to a probability vector over all of the classes
