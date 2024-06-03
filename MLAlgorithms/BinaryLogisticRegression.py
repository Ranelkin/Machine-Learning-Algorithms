import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

__author__ = "Ranel Karimov"

class BinaryLogisticRegression: 
    """
    This class implements a binary logistic regression model.
    """
    
    def __init__(self, iter=1000, lrate=0.001) -> None:
        """
        Initialize the binary logistic regression model.

        Parameters:
        iter (int): The number of iterations for the gradient descent algorithm.
        lrate (float): The learning rate for the gradient descent algorithm.
        """
        self.iter = iter
        self.lrate = lrate
        self.w = None  # weights of the model
        self.loss_hist = np.empty(self.iter)  # history of loss values
        
    def sig(self, a):
        """
        Compute the sigmoid function.

        Parameters:
        a (float): The input to the sigmoid function.

        Returns:
        float: The output of the sigmoid function.
        """
        return 1/(1+np.exp(-a))

    def Loss(self, X, y, p): 
        """
        Compute the loss function for the binary logistic regression model.

        Parameters:
        X (numpy.ndarray): The feature matrix.
        y (numpy.ndarray): The target vector.
        p (float): The probability threshold for classification.

        Returns:
        float: The value of the loss function.
        """
        return -np.sum(y*np.log(p) + (1 - y)*np.log(1 - p)) * 1/X.shape[0]
    
    def fit(self, X, y) -> None:
        """
        Fit the binary logistic regression model to the data.

        Args:
            X (numpy.ndarray): The feature matrix.
            y (numpy.ndarray): The target vector.
        """
        # Extend dimension for bias 
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Initialize weights with small random numbers
        self.w = np.random.randn(X.shape[1])
        
        for epoch in range(self.iter): 
            p = self.sig(X @ self.w)  # Calculate the probability
            L = self.Loss(X, y, p)  # Calculate the loss
            grad = (p - y) @ X  # Calculate the gradient
            self.w -= self.lrate * grad  # Update the weights
            self.loss_hist[epoch] = L  # Store the loss value

    def predict(self, X) -> np.ndarray:
        """
        Predict the target values for the given feature matrix.

        Args:
            X (numpy.ndarray): The feature matrix.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        threshold = 0.5
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Extend dimension for bias
        probabilities = self.sig(X @ self.w)  # Calculate the probabilities
        return (probabilities >= threshold).astype(int)  # Return the predicted target values


if __name__ == '__main__':
    iris = load_iris ()
    # Store observations.
    X = iris.data
    y = iris.target
    # Show classes.
    print(*iris.target_names)
    print(*iris.feature_names)
    # Select first two features.
    X = X[:,:2]
    # Show matrix shape.
    print(X.shape)
    # Get total number of samples.
    num_samples , _ = X.shape
    # Compute number of samples for training.
    num_train = num_samples * 70 // 100
    # Compute number of samples for training.
    num_test = num_samples - num_train
    # Adjust classes for binary classification.
    y_bin = np.copy(y) 
    y_bin[y > 0] = 1
    # Compute random ordering.
    indices = np.random.permutation(num_samples)
    # Define training set.
    X_train = X[indices[:num_train]] 
    y_train = y_bin[indices[:num_train]]
    # Define test set.
    X_test = X[indices[num_train:]] 
    y_test = y_bin[indices[num_train:]]
    # Create classifier.
    model = BinaryLogisticRegression()
    # Train the model with gradient descent.
    loss_history = model.fit(X_train , y_train) # Get predictions for test data.
    y_pred = model.predict(X_test) # Compute accuracy on test set.
    plt.plot([i for i in range(1000)], model.loss_hist)
    plt.show()
    acc = np.sum(y_test == y_pred)/num_test*100 
    print(f'Test Accuracy: {acc:.2f}%')