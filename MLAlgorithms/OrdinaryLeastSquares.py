from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt


__author__ = 'Ranel Karimov'

class OrdinaryLeastSquares:
    
    def __init__(self):
        self.wmle = None 
        
        
    def fit(self, X, Y) -> None:
        #Extend dimension by 1 for bias term.
        X = np.array(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        #Tranpose the matrix
        X_T= X.T

        #Calculate the weights
        weights = np.linalg.pinv(X_T @ X) @ (X_T @ Y)
        self.wmle = weights
        
        
    def predict(self, X) -> list:
    
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.wmle 

    
if __name__ == '__main__':
    dataset = load_iris()
  
    
    #Split the data into train and test set
    split = train_test_split(dataset.data, test_size=100, train_size=50)
    
    #Split the data into features and target
    X_train = split[0][:, :2]  
    y_train1 = np.concatenate(split[0][:, -2:-1], axis=0) 
    y_train2 = np.concatenate(split[0][:, -1:], axis=0)
    # Testing data
    X_test = split[1][:, :2]  
    y_test1 = np.concatenate(split[1][:, -2:-1], axis=0)
    y_test2 = np.concatenate(split[1][:, -1:], axis=0)
     
    ols = OrdinaryLeastSquares()
    #Fit the model
    ols.fit(X_train, y_train1)
    #Predict values for petal length
    y_pred1 = ols.predict(X_test)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Plot the results
    #Predicted 
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred1, color='red', label='Predicted petal length')
    #Actual
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test1, color='blue', label='Actual petal length')
    
    ols.fit(X_train, y_train2)
    #Predict values for petal length
    y_pred2 = ols.predict(X_test)
    
    #Plot the results
    #Predicted 
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred2, color='orange', label='Predicted petal width')
    #Actual
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test2, color='black', label='Actual petal width')
    
    
    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('sepal width (cm)')
    ax.set_zlabel('Prediction')
    ax.legend()
    
    plt.show()