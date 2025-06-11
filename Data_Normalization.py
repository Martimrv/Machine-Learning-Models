import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
This code is used to normalize the datasets.
"""

# Read the dataset
df = pd.read_csv('IrisDataset.csv')

# Separate features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
species = df['species']

# Create and fit scaler for features
scaler = MinMaxScaler()

# Normalize features
X_normalized = scaler.fit_transform(X).round(2)

# Create normalized DataFrame with original column names
df_normalized = pd.DataFrame(X_normalized, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

# Add back the species column
df_normalized['species'] = species

# Save the normalized dataset
df_normalized.to_csv('IrisDataset_normalized.csv', index=False)
print("Dataset has been normalized and saved as 'IrisDataset_normalized.csv'") 