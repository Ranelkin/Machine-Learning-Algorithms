from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

__author__ = "Ranel Karimov"

class SupportVectorMachineImp(): 
    """
    This class implements a Support Vector Machine model.
    """
    
    def __init__(self) -> None:
        """
        Initialize the Support Vector Machine model.
        """
        self.C = None  # Regularization parameter
        self.gamma = None  # Kernel coefficient
        self.svm = None  # The trained SVM model
        
    def cross_validation(self, X, y):
        """
        Perform cross-validation and grid search to find the best parameters for the SVM model.

        Args:
            X (numpy.ndarray): The feature matrix.
            y (numpy.ndarray): The target vector.
        """
        # Parameters for cross validation and grid search
        param_grid = {"kernel": ["rbf"], "C": [0.001, 0.01, 0.1, 1, 10], "gamma": [0.01, 0.1, 1, 10]}

        # Perform grid search
        grid_search = GridSearchCV(SVC(), param_grid, scoring="accuracy")
        grid_search.fit(X, y)
        results = pd.DataFrame(grid_search.cv_results_)
        
        # Print the results
        print(results)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best estimator: {grid_search.best_estimator_}")
        print(f"Achieved score: {grid_search.best_score_}")
        
        # Store the best parameters
        self.C = grid_search.best_params_["C"]
        self.gamma = grid_search.best_params_["gamma"]
        
        # Generate a heatmap of the results
        print("Generating Heatmap...")
        pivot_table = results.pivot_table(values='mean_test_score', index='param_gamma', columns='param_C')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title('Grid Search CV Results')
        plt.ylabel('gamma')
        plt.xlabel('C')
        plt.show()
    
    def fit(self, X, y): 
        """
        Fit the SVM model to the data.

        Args:
            X (numpy.ndarray): The feature matrix.
            y (numpy.ndarray): The target vector.
        """
        svm = SVC(C=self.C, kernel="rbf", gamma=self.gamma)
        self.svm = svm.fit(X, y)
        
    def test_svm_implementation(self, X, y): 
        """
        Test the SVM model on the given data and print the accuracy score.

        Args:
            X (numpy.ndarray): The feature matrix.
            y (numpy.ndarray): The target vector.
        """
        prediction = self.svm.predict(X)
        print(f"Accuracy score: {accuracy_score(y, prediction)}")
        
     
if __name__ == '__main__': 
    # Load the iris dataset and split it into a training set and a test set
    dataset = load_iris()
    features = dataset.data
    target = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, train_size=0.7)
    
    # Train and test the SVM model
    model = SupportVectorMachineImp()
    model.cross_validation(features, target)
    model.fit(X_train, y_train)
    model.test_svm_implementation(X_test, y_test)