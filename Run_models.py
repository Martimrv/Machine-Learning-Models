import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MachineLearningModel import KNNClassificationModel, KNNRegressionModel
from sklearn.model_selection import train_test_split
from itertools import combinations

"""
This file is used to run the KNN classification and Regression models easily.
"""

def run_classification():
    print("\n=== Running KNN Classification ===")
    df = pd.read_csv('IrisDataset_normalized.csv')
    X = df[['PetalLengthCm', 'PetalWidthCm']].values
    y = df['species']
    
    # Test different k values
    k_values = [3, 5, 7, 9]
    for k in k_values:
        print(f"\nTesting k={k}")
        model = KNNClassificationModel(k=k)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate accuracy
        correct_predictions = model.evaluate(y, predictions)
        print(f"Accuracy: {correct_predictions}")
        
        # Create decision boundary visualization
        fig, ax = plt.subplots()
        model.knn_mesh_decision_boundary(X, y, ax)
        plt.title(f'KNN Decision Boundary (k={k})')
        plt.savefig(f'knn_decision_boundary_k{k}.png')
        plt.show()  

def run_regression():
    """
    This function is for Problem 1.
    """
    print("\n=== Running KNN Regression ===")
    # Load the polynomial dataset
    polynomial = pd.read_csv('Polynomial200_normalized.csv')
    polynomial_X = polynomial['x']
    polynomial_y = polynomial['y']
    
    # Test different k values
    accuracies = []
    k_values = [3, 5, 7, 9]
    for k in k_values:
        print(f"\nTesting k={k}")
        model = KNNRegressionModel(k=k)
        model.fit(polynomial_X, polynomial_y)
        predictions = model.predict(polynomial_X)
        mse = model.evaluate(polynomial_y, predictions)
        accuracies.append(mse)
        # Create regression plot
        model.knn_regression_plot(polynomial_X, polynomial_y)
    print("****** KNN Regression Model MSE Values ******")
    for k, value in zip(k_values, accuracies):
        print(f"k={k}: {value:.4f}")

def run_regression_experiment():
    """
    Run the KNN regression with:
    - 10 repetitions
    - k values: 3, 5, 7, 9, 11, 13, 15
    - 80-20 train-test split
    - Different random seeds for each repetition
    """
    print("\n=== Running KNN Regression Problem 2 ===")
    
    # Load and prepare the polynomial dataset
    polynomial = pd.read_csv('Polynomial200_normalized.csv')
    X = polynomial['x'].values
    y = polynomial['y'].values
    
    # Requirements 
    k_values = [3, 5, 7, 9, 11, 13, 15]
    n_repetitions = 10
    test_size = 0.2
    random_seeds = range(42, 42 + n_repetitions)  # 10 different seeds
    
    results = np.zeros((n_repetitions, len(k_values)))
    
    for rep_idx, seed in enumerate(random_seeds):
        print(f"\nRepetition {rep_idx + 1}/10")
        
        # Split data with current seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        
        # Test each k value
        for k_idx, k in enumerate(k_values):
            print(f"Testing k={k}")
            model = KNNRegressionModel(k=k)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = model.evaluate(y_test, predictions)
            results[rep_idx, k_idx] = mse
            print(f"MSE for k={k}: {mse:.6f}")
    
    # Calculate mean and std of MSE for each k
    mean_mse = np.mean(results, axis=0)
    std_mse = np.std(results, axis=0)
    
    # Create Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(k_values))
    plt.bar(x, mean_mse, yerr=std_mse, capsize=5)
    plt.xlabel('k value')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('KNN Regression BarChart')
    plt.xticks(x, k_values)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(mean_mse):
        plt.text(i, v + std_mse[i], f'{v:.6f}', ha='center', va='bottom')
    
    plt.savefig('KNNRegressionBarChart.png')
    plt.close()
    
    print("\n===== Results =====")
    print("\nMean MSE for each k value:")
    for k, mean, std in zip(k_values, mean_mse, std_mse):
        print(f"k={k}: {mean:.6f} ± {std:.6f}")
    
    best_k_idx = np.argmin(mean_mse)
    best_k = k_values[best_k_idx]
    print(f"\nBest k value: {best_k} (MSE: {mean_mse[best_k_idx]:.6f} ± {std_mse[best_k_idx]:.6f})")
    print(f"- Worst k value: {k_values[np.argmax(mean_mse)]} (MSE: {np.max(mean_mse):.6f})")

def find_best_features_classification():
    """
    Function to find the best combination of two features for the Iris dataset classification.
    Tests all possible pairs of features with different k values.
    Creates visualization for each combination.
    """
    print("\n=== Finding Best Feature Combination for KNN Classification ===")

    # Load and prepare data
    df = pd.read_csv('IrisDataset_normalized.csv')
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    y = np.array(df['species'])  # Convert to numpy array
    
    # Automatically generate all possible pairs of features
    feature_pairs = list(combinations(features, 2))
    
    k_values = [3, 5, 7, 9]
    n_splits = 5  
    results = {}  
    
    for pair in feature_pairs:
        print(f"\nTesting feature pair: {pair}")
        X = df[list(pair)].values  # Convert to numpy array
        pair_results = np.zeros((len(k_values), n_splits))
        
        for split_idx in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42 + split_idx
            )
            
            for k_idx, k in enumerate(k_values):
                model = KNNClassificationModel(k=k)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = model.evaluate(y_test, predictions) / len(y_test)
                pair_results[k_idx, split_idx] = accuracy
        
        results[pair] = np.mean(pair_results, axis=1)
        
        # Create scatter plot for this feature pair
        plt.figure(figsize=(10, 6))
        unique_species = np.unique(y)
        colors = ['blue', 'red', 'green']
        
        for species, color in zip(unique_species, colors):
            mask = df['species'] == species
            plt.scatter(
                df[pair[0]][mask], 
                df[pair[1]][mask],
                c=color,
                label=species,
                alpha=0.6
            )
        
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.title(f'Scatter Plot: {pair[0]} vs {pair[1]}\nBest Accuracy: {np.max(results[pair]):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'scatter_{pair[0]}_{pair[1]}.png')
        plt.close()
    
    # Find best combination
    best_pair = max(results.items(), key=lambda x: np.max(x[1]))
    best_k_idx = np.argmax(best_pair[1])
    best_k = k_values[best_k_idx]
    
    print("\n===== Results =====")
    print("\nAccuracy for each feature combination:")
    for pair, accuracies in results.items():
        print(f"\n{pair}:")
        for k, acc in zip(k_values, accuracies):
            print(f"k={k}: {acc:.4f}")
    
    print(f"\nBest combination:")
    print(f"Features: {best_pair[0]}")
    print(f"k value: {best_k}")
    print(f"Accuracy: {best_pair[1][best_k_idx]:.4f}")
    
    # Summary Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(feature_pairs))
    
    # Plot average accuracy for each pair
    avg_accuracies = [np.max(results[pair]) for pair in feature_pairs]
    bars = plt.bar(x, avg_accuracies)
    
    # Customize the plot
    plt.xlabel('Feature Pairs')
    plt.ylabel('Best Accuracy')
    plt.title('Best Accuracy for Different Feature Pairs')
    plt.xticks(x, [f'{p[0]}\n{p[1]}' for p in feature_pairs], rotation=45)
    plt.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars, avg_accuracies):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{acc:.4f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig('feature_pairs_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Run both models
    #run_classification()
    #run_regression()
    #run_regression_experiment()
    find_best_features_classification() 