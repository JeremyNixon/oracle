import random
import numpy as np
import pandas as pd

def k_means_plus_plus(x_train, k):
    old_means = x_train[np.random.choice(len(x_train), k)]

    # k-means ++ initialization
    x_train[start]
    new_means = []
    new_means.append(x_train[start])
    for iteration in xrange(k-1):
        distances = []
        for index, datapoint in enumerate(x_train):
            total_distances = 0
            for mean in new_means:
                total_distances += np.sqrt(np.sum((mean-datapoint)**2))
            distances.append(total_distances)
        distances /= sum(distances)
        new_means.append(x_train[np.random.choice(len(x_train), 1, p=distances)])
        
    new_means = np.array(new_means)
    while(not np.array_equal(old_means, new_means)):
        old_means = new_means
        assignments = {}
        assignment_index = {}
        # compute responsibilities
        for index, datapoint in enumerate(x_train):
            # Compute distance to each mean
            distances = [[i, np.sqrt(np.sum((datapoint-mean)**2))] for i, mean in enumerate(new_means)]
            distances.sort(key = lambda x: x[1])
            # assign each datapoint to closest mean
            if distances[0][0] in assignments:
                assignments[distances[0][0]].append(datapoint)
                assignment_index[distances[0][0]].append(index)
            else:
                assignments[distances[0][0]] = [datapoint]
                assignment_index[distances[0][0]] = [index]
        # compute new means
        new_means = np.array([np.mean(assignments[index], axis=0) for index in assignments.keys()])            
    return assignment_index, new_means