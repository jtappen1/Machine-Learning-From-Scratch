# Linear Classifier

## Goal:
The goal of this excercise is to learn and understand the math around gradient descent and back propagation, as well as create a model with no outside help.

## Steps:
### Step 1: 
Softmax: $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$  

Cross Entropy Loss: $\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$  

### Step 2:
We are calculating the loss $L$ w.r.t $w$ or $\frac{\partial L}{\partial w}$

Set up the derivation of $\frac{\partial L}{\partial w}$ through the use of the chain rule.

$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$  

We use this formula to calculate the gradient vector.  This tells us the direction and magnitude that we should adjust our weights and bias by.


### Step 3:
Compute the individual derivatives. Start with Cross Entropy Loss  
  
For Cross Entropy loss, the function takes $y$, a one-hot encoded vector of shape [batch_size, num_classes]  
and $\hat{y}$, the vector of predicted class scores in shape [batch_size, num_classes].  

Something important to remember is that y is a one-hot encoded vector.  Which means in the vector y, there will only be a single values that is 1 denoting the correct class, everything else in the row will be 0. EX: [0,1,0,0,0] correct class is at index 1.  This means we can simplify the expression to
  
$\frac{\partial L}{\partial \hat{y}} = - \log{\hat{y_j}}$


We then take the overall derivative of the function  
$\frac{\partial L}{\partial \hat{y_j}} = - y_j \cdot \frac{1}{\hat{y_j}} = -\frac{y_j}{\hat{y_j}}$  

Writing that in vector notation gives us 
$\frac{\partial L}{\partial \hat{\vec{y}}} =  -\frac{\vec{y}}{\hat{\vec{y}}}$  

### Step 4:
Compute the derivative of $\hat{y}$ w.r.t $z$  
$\hat{y}$ is the output from the softmax, so we are taking the derivative of the softmax with respect to $z$  
The softmax derivative is given by: $\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i (1 - \hat{y}_i)$ if $i = j$, and $-\hat{y}_i \hat{y}_j$ if $i \ne j$.

### Step 5:
Compute the derivative of $z$ w.r.t $w$  
we know that $z = X \cdot W + b$, but when we take the derivative the $b$ is a constant and goes away.
$\frac{\partial z}{\partial w} = x^T$

### Step 6:
We then go through and multiply every parital derivative together.  We end up with  
$\frac{\partial L}{\partial z} = \hat{y} - y$  
$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x^T$   
$\frac{\partial L}{\partial b} = (\hat{y} - y)$  

We then use these derivatives to update the weights and bias's with the respective gradient vectors.


