from abc import ABC, abstractmethod
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

class KNNRegressionModel(MachineLearningModel):
    """
    Class for KNN regression model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, point, data):
        """
        Calculate the Euclidean distance between a point and the dataset points.
        Euclidean equation: sqrt(sum((X₂-X₁)²)) where:
        X₂ = New entry's data point
        X₁ = Existing entry's data point
        """
        point = np.array(point).reshape(1, -1)  # Ensure 2D array with shape (1, n_features)
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)  # Reshape 1D array to 2D
        return np.sqrt(np.sum((point - data) ** 2, axis=1))
    
    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.X_train = np.array(X)
        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by averaging the target variable of the k nearest neighbors.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        predictions = []
        for row in X:
            distances = self.euclidean_distance(row, self.X_train)
            sorted_distances = np.argsort(distances)
            top_k_rows = sorted_distances[:self.k]
            mean_value = np.mean(self.y_train[top_k_rows])
            predictions.append(mean_value)
        return np.array(predictions)
        

    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the Mean Squared Error (MSE) between the true and predicted values.
        The MSE is calculated as the average of the squared differences between the true and predicted values.        

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        return np.mean((np.array(y_true) - np.array(y_predicted)) ** 2)

    def knn_regression_plot(self, X, y):
        """
        Create a visualization of the KNN regression model
        """
        # Convert inputs to numpy arrays 
        X = np.array(X)
        y = np.array(y)
        plt.figure(figsize=(10, 6))
        
        # Plot the original data points
        plt.scatter(X, y, color='blue', alpha=0.5, label='Original Data Points')
        # Create a range of points for prediction
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        # Make predictions for the range of points
        y_pred = self.predict(X_range)
        
        # Plot the regression line
        plt.plot(X_range, y_pred, color='red', label=f'KNN Regression (k={self.k})')
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'KNN Regression with k={self.k}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'KNNRegression_Polynomial_{self.k}.png')
        plt.close()
        
        print(f"Plot has been saved as 'KNNRegression_Polynomial_{self.k}.png'")

class KNNClassificationModel(MachineLearningModel):
    """
    Class for KNN classification model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def most_common_rows(self, lst):
        """
        Get most frequent class from a list of labels.
        """
        # Convert numpy arrays to list of strings
        return max(set(lst), key=list(lst).count)

    def euclidean_distance(self, point, data):
        """
        Calculate the Euclidean distance between a point and the dataset points.
        Euclidean equation: sqrt((X₂-X₁)²+(Y₂-Y₁)²) where:
        X₂ = New entry's data.
        X₁= Existing entry's data.
        Y₂ = New entry's data.
        Y₁ = Existing entry's data."""
        
        point = np.array(point)
        data = np.array(data)
        return np.sqrt(np.sum((point - data) ** 2, axis=1))
    
    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by taking the mode (majority) of the target variable of the k nearest neighbors.
        
        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        predictions = []
        X = np.array(X)
        for row in X:
            distances = self.euclidean_distance(row, self.X_train)
            sorted_distances = np.argsort(distances)
            top_k_rows = sorted_distances[:self.k]
            neighbors = self.y_train[top_k_rows]
            predictions.append(self.most_common_rows(neighbors))
        return predictions
        

    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the total number of correct predictions only.
        Do not use any other evaluation metric.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        # Convert inputs to numpy arrays to ensure consistent handling
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        
        correct_predictions = 0
        for i in range(len(y_true)):
            if y_true[i] == y_predicted[i]:
                correct_predictions += 1
        return correct_predictions
    
    def knn_mesh_decision_boundary(self, X, y, ax):
        """
        Create a visualization of the KNN classification model
        """
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z_encoded = label_encoder.transform(Z)
        Z_encoded = Z_encoded.reshape(xx.shape)
    
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z_encoded, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y_encoded, edgecolor='k', s=30, cmap=plt.cm.coolwarm)
        plt.title("Mesh decision boundary for the Iris2D dataset using 5-NN")
        plt.xlabel("Sepal length (cm)")
        plt.ylabel("Sepal width (cm)")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=plt.cm.coolwarm(i / 2), label=cls) for i, cls in enumerate(label_encoder.classes_)]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.grid(True)
        plt.show()