import numpy as np
import random

def kmeans(x_train, k):
    # Initialize means to random datapoints
    oldmeans = [x_train[i] for i in [np.random.randint(len(x_train)) for i in range(k)]]
    means = [x_train[i] for i in [np.random.randint(len(x_train)) for i in range(k)]]
    
    # Iterate until assignments of datapoints to means is the same between iterations
    while(([(oldmeans[i] == means[i]).all() for i in xrange(k)]) != [True for i in xrange(k)]):
        oldmeans = means
        assignments = {}
        
        # Assign datapoints to closest mean
        for i, datapoint in enumerate(x_train):
            distance = float("inf")
            for index, mean in enumerate(means):
                d = sum((datapoint - mean)**2)
                if d < distance:
                    distance = d
                    cluster = index
            try:
                assignments[cluster].append(i)
            except:
                assignments[cluster] = [i]                    
                    
        # Update means to be the average of the assigned datapoints
        for i in xrange(k):
            means[i] = sum([x_train[data_index] for data_index in assignments[i]])/len(assignments[i])
    
    return assignments