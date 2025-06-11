import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing

"""
This class is used to provide data visualization.
"""

class DataVisualization:
    def __init__(self, df=None):
        self.df = df

    def load_iris_datasets(self):
        df = pd.read_csv('IrisDataset_normalized.csv')
        return df
    
    def load_polynomial_datasets(self):
        df = pd.read_csv('Polynomial200_normalized.csv')
        return df

    def preview_iris_dataset(self):
        df = self.load_iris_datasets()
        print(df.head())
        print(df.describe())
        print(df['species'].value_counts())
    
    def preview_polynomial_dataset(self):
        df = self.load_polynomial_datasets()
        print(df.head())
        print(df.describe())
        print(df['y'].value_counts())
        print(df['x'].value_counts())
        
    def plot_iris_dataset(self):
        df = self.load_iris_datasets()
        graph = sns.pairplot(df, hue='species', markers=['x'])
        plt.show()
    
    def plot_polynomial_dataset(self):
        df = self.load_polynomial_datasets()
        graph = sns.scatterplot(x='x', y='y', data=df)
        plt.show()
    
if __name__ == "__main__":
    data_visualization = DataVisualization()
    data_visualization.preview_iris_dataset()
    data_visualization.preview_polynomial_dataset()
    # Plots are commented just to avoid constantly opening the plots.
    #data_visualization.plot_iris_dataset()
    #data_visualization.plot_polynomial_dataset()
    
