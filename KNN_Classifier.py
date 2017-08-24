import math
import numpy as np
import pandas as pd
import sklearn.cross_validation

def knn(x_train, y_train, x_test, k):
    predictions = []
    # Compute solution for each test datapoint
    for datapoint in x_test:
        distances = []
        
        # Get distance to every training set datapoint
        for index, vector in enumerate(x_train):
            distances.append([np.sum(np.sqrt((datapoint-vector)**2)), y_train[index]])
                              
        # Sort by distance
        distances.sort()

        # Plurality prediction
        counts = {}
        for i in xrange(k):
            try:
                counts[distances[i][1]] += 1
            except:
                counts[distances[i][1]] = 1
        
        # Recover Maximum
        base = 0
        for i in xrange(len(np.unique(y_train))):
            try:
                value = counts[i]
                if value > base:
                    prediction = i
            except:
                pass
        # Add Maximum to Predictions
        predictions.append(prediction)
    return predictions        

iris = pd.read_csv('/Users/jeremynixon/Dropbox/python/Algorithms/practice/iris.data', header=None)
y = iris[4]
iris = iris.drop([4], 1)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(iris, y, test_size = .20, random_state=42)

predictions = knn(x_train, y_train, x_test, 4)
print predictions