import numpy as np
import pandas as pd

class KNNRegressor():
    def __init__(self, k=4):
        self.k = k
        self.x_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        
    def predict(self, x_test):
        predictions = []
        for i, test in enumerate(np.array(x_test)):
            # For each test datapoint, compute distance to each training datapoint
            distances = [(np.sum(np.abs(test-j)), index) for index, j in enumerate(self.x_train)]
            distances.sort()
            
            # Grab the indices of the closest k training datapoints
            indices = np.array(distances)[:, 1][:self.k]
            
            # Predict the average of the closest datapoints
            predictions.append(np.mean(self.y_train[np.array(indices, int)]))
        return predictions

iris = pd.read_csv('/Users/jeremynixon/Dropbox/python_new/oracle/data/iris.csv')
labels = iris['sepal_length']
features = iris.drop(['sepal_length'], 1)

import sklearn.model_selection
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size = .20, random_state=42)

knn = KNNRegressor()
knn.fit(x_train, y_train)
preds = knn.predict(x_test)
for i in zip(preds, y_test):
    print i     