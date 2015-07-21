import math
import numpy as np

def minkowski_distance(sample1, sample2, dimensions, q = 2):
    """
    We will default to Eucledian Distance (q = 2), change q in order to 
    adapt to a different distance metric.
    """
    
    distance = 0
    # Iterate over each dimension and add the difference to the sum.
    for dimension in range(dimensions):
        distance += abs(sample1[dimension] - sample2[dimension])**q
    
    minkowski_distance = distance**(1.0/q)
    
    return minkowski_distance

def neighbors(x_train, test_datapoint, k, q=2):
    # Calculate the distance from our test datapoint to every training datapoint.
    distances = []
    for datapoint in range(len(x_train)):
        distance = minkowski_distance(x_train[datapoint], test_datapoint, len(test_datapoint)-1, q)
        distances.append((x_train[datapoint], distance, datapoint))
    
    # Sort training datapoints based on distance
    distances.sort(key=lambda x: x[1])
    
    neighbors = []
    for neighbor in range(k):
        neighbors.append(distances[neighbor][2])
        
    return neighbors

def outcome(neighbors, y_train):
    # Check the classes of the nearest neighbors, and choose the class that is most prevalent
    neighbor_classes = {}
    for neighbor in range(len(neighbors)):
        response = y_train[neighbors[neighbor]].item(0)
        if response in neighbor_classes:
            neighbor_classes[response] += 1
        else:
            neighbor_classes[response] = 1
    votes = sorted(neighbor_classes.iteritems(), key = lambda x: x[1])
    return votes[0][0]

def KNN_Classifier(x_train, y_train, x_test, k, q = 2):	

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    predictions = []
    for i in range(len(x_test)):
        nearest_neighbors = neighbors(x_train, x_test[i], k, q)
        result = outcome(nearest_neighbors, y_train)
        predictions.append(result)
    return predictions