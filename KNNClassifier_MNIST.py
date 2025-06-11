from keras.datasets import mnist
from matplotlib import pyplot
from FastKNNClassificationModel import FastKNNClassificationModel
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print("X_train shape:", train_X.shape)
print("y_train shape:", train_y.shape)
print("X_test shape:", test_X.shape)
print("y_test shape:", test_y.shape)

#for i in range(9):  
 #   pyplot.subplot(330 + 1 + i)
  #  pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
   # pyplot.show()

X_train_falt_img = train_X.reshape(train_X.shape[0], -1)
X_test_falt_img = test_X.reshape(test_X.shape[0], -1)

X_train_normalized = X_train_falt_img / 255.0
X_test_normalized = X_test_falt_img / 255.0

class KNNClassifier_MNIST:
    def __init__(self, k=3):
        self.k = k
        self.model = FastKNNClassificationModel(k=self.k)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def main(self):
        # Smaller subset
        train = 10000
        test = 1000
        
        k = [3, 5, 7, 9, 11]
        # find best k value
        for k in k:
            knn = KNNClassifier_MNIST(k=k)
            print(f"\nTesting with k={k}")
            print("Training on subset with ", train, " samples")
            knn.fit(X_train_normalized[:train], train_y[:train])
            
            print("Testing on ", test, " samples")
            predictions = knn.predict(X_test_normalized[:test])
            
            accuracy = knn.evaluate(test_y[:test], predictions) / test
            print(f"Accuracy with k={k}: {accuracy:.4f}")
            print("-" * 50)

if __name__ == "__main__":
    knn = KNNClassifier_MNIST()
    knn.main()


