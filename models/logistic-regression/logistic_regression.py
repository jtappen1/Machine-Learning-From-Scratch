import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.001, epochs = 1000):
        self._lr = lr
        self._epochs = epochs
        self._weights = None
        self._bias = None
        self._losses = []


    def sigmoid(self, x: float) -> float: 
        """
        Sigmoid activation function: 1 / (1 + e^-x)
        """
        return 1 / (1 + np.exp(-x))
    
    def compute_bce_loss(self, x_true: float, x_pred: float) -> float:
        """
        Compute BCE Loss  -1/n (sum(0,N) -> [x * log(x^) + (1-x) * log(1 - x^)])
        """
        # We add a small value (epsilon) because log(0) is undefined
        epsilon = 1e-9
        x1 = x_true * np.log(x_pred + epsilon)
        x2 = (1 - x_true) * np.log(1-x_pred + epsilon)
        return  - np.mean(x1 + x2)
    
    def feed_forward(self, x):
        """
        Runs the feed forward step.  This happens on every step, and sets up a single step of gradient descent.
        In the forward pass, the model takes an input and computes the predicted output.
        The steps are as follows:
            - Apply transformations to the input, layer by layer for each layer in the model.  This includes
                -  Multiply by weights
                -  Add bias
                -  Apply activation function
            - In our case, z = x(input) * w (weights) + b (bias) and out activation function is Sigmoid, 
                which will compress our output values between 0 and 1.
        """
        # First, apply transformations to the input.  
        # Np.dot does the dot product if 1D arrays or matrix multiplication if multi-dimensional
        z = np.dot(x, self._weights) + self._bias

        # Apply the activation function to our predicted output
        return self.sigmoid(z)
    
    def fit(self, X: np.array, y):
        """
        This function defines the process of fitting the data to the logistic regression model.
        Args:
            - X: Input data
            - y: Labels

        """
        # X is our input array of data.
        # n_samples is the number of samples in the array of data.  
        # n_features is the number of data points in each instance of data.
        # EX:
        #   X = [
        #       [5.1, 3.5, 1.4, 0.2],  # sample 1
        #       [4.9, 3.0, 1.4, 0.2],  # sample 2
        #       ]
        # n_samples = 2,
        # n_features = 4
        n_samples, n_features = X.shape

        # Initialize the weights, according to the number of features.  
        # These are the internal values of the model that we will be changing.
        # We set # of weights equal to the # of features because we want each individual input feature 
        # to be tuned and adjusted by model training.
        self._weights = np.zeros(n_features)

        self._bias = 0

        # This loop comprises gradient descent. 
        for _ in range(self._epochs):
            # First we predict the output values by running feed forward on X
            # We run the feed forward on the entirety of the input data

            # The length of the feed forward loop is the lengths of the number of samples
            pred_output = self.feed_forward(X) # returns a vector of predicted probabilities.

            # We get the predicted output, and computes the loss based off the true and predicted values.
            # Returns a single value for the loss
            self._losses.append(self.compute_bce_loss(y,pred_output))
            
           # We next need to update the weights and biases, to help us find the 
           # correct direction to proceed in each dimension

           # Backpropagation:
           # We want to compute gradients of the loss w.r.t. weights and bias
            dz = pred_output - y # dz is the array of the 1s and 0s - the true values

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            # we compute the average of the error of all predictions
            # the sum of all errors, the pred output - the actual values.  
            db = (1 / n_samples) * np.sum(dz)
            # update parameters
            # we follow the update rule.  So we update the weights with the learning rate * 
            # the derivative of the weights or bias
            # we subtract and are trying to minimize the loss 
            self._weights -= self._lr * dw
            self._bias -= self._lr * db

    def predict(self, X):
        # We start with the same array of images with 30 features
        threshold = 0.5
        # We go compute the dot product of each sample with the weights of the model, and then add the bias to each one.
        y_hat = np.dot(X, self._weights) + self._bias
        # Apply the sigmoid funciton to each sample, forcing it between 0 and 1
        y_predicted = self.sigmoid(y_hat)
        # If greater than the threshold, it is a 1 else 0
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]
        
        return np.array(y_predicted_cls)



