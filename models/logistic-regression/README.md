# Logistic Regression

# Backprop and Gradient Descent
- We are calculating the gradients of the Loss with respect to the weights
- $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$

We first compute the derivative of BCE Loss, or the Loss $L$ with respect to the $\hat {y}$ vector.  
- This is because we are passing the predicted values into the sigmoid function. 
- The $\hat {y}$ vector is the direct output of our model.  This is what is passed into the Loss function.  
- So we take the derivative of essentially both parts to make our way backwards.

vectors are list of numbers, they represent an offset that have both magnitude and direction

This is the overall chain rule showing the partial derivatives.
- $\frac{\partial L}{\partial w}$ = $\frac{\partial L}{\partial \hat {y}} \cdot \frac{\partial \hat {y}}{\partial z} \cdot \frac{\partial z}{\partial w}$ 

So initally we have the sigmoid, which computes the values for our $\hat {y}$
- $\sigma(x) = \frac{1}{1 + e^{-x}}$

And BCE which looks like 
- $\mathcal{L}(\hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$


$\hat{y} = \sigma(w_0 \cdot x_0 + w_1 \cdot x_1 + w_2)$

<!-- $\frac{\partial L}{\partial w_0}=\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_0}$  -->
This line denotes the update step of any machine learing model.  We are updating the weights to minimize the loss function.  The gradient given by $\frac{\partial L}{\partial w}$ tells us the direction we need to go, as well as how much, and that is amplified by the learning rate which determines how big of a step.  At the end, the vector of weights themselves are updated.
- $\vec{w}_[i+1]=\vec{w}_i - \alpha \cdot \frac{\partial L}{\partial w}$


- $\frac{\partial L}{\partial \vec{w}}=\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \vec{w}} $


# Backprop:
## Step 1:
- We first compute the partial derivatives of Loss w.r.t output
- $\frac{\partial L}{\partial \hat{y}}=-[\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}]$

## Step 2
Next we compute the partial derivatives of the output w.r.t the weights/biases.  
    $\frac{\partial \hat{y}}{\partial w_0}=\frac{\partial d}{\partial dw_0}\frac{1}{1+e^{-w_0}} = (1 + e^{-x})^{-1}$ 

We do the chain rule on the function  
    $ u = 1 + e^{-x}, so (1 + e^{-x})^{-1} $ (-e^{-x})

Eventually we get  
$\frac{\partial\sigma(z)}{\partial z} =sigma(z)(1- sigma(z))= \hat{y}(1- \hat{y})$