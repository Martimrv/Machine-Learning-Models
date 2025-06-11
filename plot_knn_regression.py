import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('Polynomial200.csv')

# Create 200 equidistant points between 1 and 25
X_j = np.linspace(1, 25, 200)
Y_j = []

# For each X_j, find the 5 closest points and compute their average y-value
for x in X_j:
    # Calculate distances to all points
    distances = np.abs(df['x'] - x)
    # Get indices of 5 closest points
    closest_indices = np.argsort(distances)[:5]
    # Calculate average y-value of these 5 points
    y_avg = df.iloc[closest_indices]['y'].mean()
    Y_j.append(y_avg)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], color='blue', alpha=0.5, label='Original Data Points')
plt.plot(X_j, Y_j, color='red', label='KNN Regression (k=5)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN Regression with k=5')
plt.legend()
plt.grid(True)
plt.savefig('knn_regression_k5.png')
plt.close()

print("Plot has been saved as 'knn_regression_k5.png'") 