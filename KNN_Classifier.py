import math
import numpy as np
import operator 

def eucledian_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += (instance1[x] - instance2[x])**2
        eucledian_distance = math.sqrt(distance)
    return eucledian_distance

def get_neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = eucledian_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist, x))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][2])
    return neighbors

# Change the get_response function to allow KNN regression
def get_response(neighbors, y_train):
    class_votes = {}
    for x in range(len(neighbors)):
        response = y_train[neighbors[x]].item(0)
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]
    
def KNN_Classifier(x_train, y_train, x_test, k):

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    predictions = []
    for i in range(len(x_test)):
        neighbors = get_neighbors(x_train, x_test[i], k)
        response = get_response(neighbors, y_train)
        predictions.append(response)
    return predictions