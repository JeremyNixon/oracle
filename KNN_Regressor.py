import math
import numpy as np

def knn(x_train, y_train, x_test, k):
    predictions = []
    # Compute solution for each test datapoing
    for datapoint in x_test:
        distances = []
        
        # Get distance to every training set datapoint
        for index, vector in enumerate(x_train):
            distances.append([np.sum(np.sqrt((datapoint-vector)**2)), y_train[index]])
                              
        # Sort by distance
        distances.sort()

        # Aggregate neighboring values
        average = 0
        for i in xrange(k):
            average += distances[i][1]
        average = average / float(k)
        predictions.append(average)
    return predictions        