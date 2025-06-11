from scipy.spatial import KDTree
import numpy as np
from collections import Counter
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from MachineLearningModel import KNNClassificationModel, MachineLearningModel


class FastKNNClassificationModel(MachineLearningModel):
    """
    Fast KNN Classification Model.
    KDTree data structure for efficient nearest neighbor search.
    """
    
    def __init__(self, k):
        """
        Initialize the model with k neighbors.
        """
        self.k = k
        self.kdtree = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Build the KDTree with training data.
        """
        # Convert inputs to numpy arrays and ensure correct shape
        self.X_train = np.array(X, dtype=float)
        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)
            
        self.y_train = np.array(y)
        
        # Here we pass the training data to the KDTree data structure
        self.kdtree = KDTree(self.X_train)
    
    def predict(self, X):
        """
        Predict method for test data using KDTree's efficient nearest neighbor search.
        """
        X = np.array(X, dtype=float)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Use KDTree to find k nearest neighbors
        distances, indices = self.kdtree.query(X, k=self.k)
        
        predictions = []
        for neighbor in indices:
            neighbor_labels = self.y_train[neighbor]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            predictions.append(most_common)       
        return np.array(predictions)
    
    def evaluate(self, y_true, y_predicted):
        """
        Evaluate method by counting correct predictions.
        """
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        return np.sum(y_true == y_predicted)

class RunFastKNNClassificationModel:
    """
    Class for running the Fast KNN Classification Model.
    """
    def compare_knn_implementations():
        print("\n=== Comparing KNN Implementations ===")
 
        df = pd.read_csv('IrisDataset_normalized.csv')
        X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
        y = df['species'].values
    
        # Parameters for testing
        k_values = [1, 3, 5, 7, 9]
        n_repeats = 10 
        test_size = 0.2
        results = {
            'regular': {'times': [], 'accuracies': []},
            'fast': {'times': [], 'accuracies': []}
        }
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
        for k in k_values:
            print(f"\nTesting k={k}")
            
            # Regular KNN
            regular_times = []
            regular_model = KNNClassificationModel(k=k)
            regular_model.fit(X_train, y_train)
            
            # Time regular predictions
            for _ in range(n_repeats):
                start_time = time.time()
                predictions = regular_model.predict(X_test)
                end_time = time.time()
                regular_times.append(end_time - start_time)
            
            accuracy = regular_model.evaluate(y_test, predictions) / len(y_test)
            results['regular']['times'].append(np.mean(regular_times))
            results['regular']['accuracies'].append(accuracy)
            
            # Fast KNN
            fast_times = []
            fast_model = FastKNNClassificationModel(k=k)
            fast_model.fit(X_train, y_train)
        
            for _ in range(n_repeats):
                start_time = time.time()
                predictions = fast_model.predict(X_test)
                end_time = time.time()
                fast_times.append(end_time - start_time)
            
            # Calculate accuracy Fast
            accuracy = fast_model.evaluate(y_test, predictions) / len(y_test)
            results['fast']['times'].append(np.mean(fast_times))
            results['fast']['accuracies'].append(accuracy)
            
            for method in ['regular', 'fast']:
                print(f"{method.capitalize()} KNN - Time: {results[method]['times'][-1]:.6f}s, Accuracy: {results[method]['accuracies'][-1]:.6f}")
        
        # Create comparison plots
        plt.figure(figsize=(15, 5))
        
        # Time comparison
        plt.subplot(1, 2, 1)
        plt.plot(k_values, results['regular']['times'], 'o-', label='Regular KNN')
        plt.plot(k_values, results['fast']['times'], 'o-', label='Fast KNN')
        plt.xlabel('k value')
        plt.ylabel('Average prediction time (seconds)')
        plt.title('Execution Time Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('KNN_Models_Comparison.png')
        plt.close()
        print("\n=== Performance Summary ===")
        print("\nAverage execution times:")
        print(f"Regular KNN: {np.mean(results['regular']['times']):.6f}s")
        print(f"Fast KNN: {np.mean(results['fast']['times']):.6f}s")
        print("\nAverage accuracies:")
        print(f"Regular KNN: {np.mean(results['regular']['accuracies']):.4f}")
        print(f"Fast KNN: {np.mean(results['fast']['accuracies']):.4f}")

if __name__ == "__main__":
    RunFastKNNClassificationModel.compare_knn_implementations()
