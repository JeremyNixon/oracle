import numpy as np
import pandas as pd
import sklearn.model_selection
from collections import Counter

class KNN():
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
            
            # Predict the most common class (plurality)
            predictions.append(Counter(self.y_train[map(int, list(indices))]).most_common(1)[0][0])
        return predictions     

iris = pd.read_csv('/Users/jeremynixon/Dropbox/python_new/oracle/data/iris.csv')
y = iris['label']
x = iris.drop(['label'], 1)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = .20, random_state=42)

knn = KNN(k=5)
knn.fit(x_train, y_train)
preds = knn.predict(x_test)
accuracy = Counter(preds-y_test)[0]/float(len(y_test))
print accuracy, preds