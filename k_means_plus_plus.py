import random
import numpy as np
import pandas as pd


def L2_norm(vector):
    count = 0
    for i in vector:
        i = i**2
        count += i
    norm = np.sqrt(count)
    return norm


def k_means_plus_plus_initialize(x_train, k):
    means = []
    original_mean = list(np.array(random.sample(x_train, 1))[0])
    means.append(original_mean)

    for clusters in range(k-1):
        
        distances = []
        for i in range(len(x_train)):
            distance_i = 0
            for mean in means:
                distance_i += L2_norm(x_train[i]-mean[0])
            distances.append([x_train[i], distance_i])

        count = 0
        for i in distances:
            count += i[1]

        for i in range(len(distances)):
            distances[i][1] = distances[i][1]/count

        for i in range(len(distances)):
            if i == 0:
                distances[i][1] = distances[i][1]
            else:
                distances[i][1] = distances[i-1][1] + distances[i][1]

        rand = random.uniform(0,1)

        for i in range(len(distances)):
            if rand < distances[i][1]:
                choice = i-1
                break

        new_mean = np.array(x_train[choice])
        means.append(list(new_mean))

    return means


def cluster(x_train, means):
    clusters = {}
    # Assign each datapiont to a cluster based on L2 distance to cluster center
    for datapoint in x_train:
        best_mean_key = min([(i[0], L2_norm(datapoint-means[i[0]])) \
                        for i in enumerate(means)], key=lambda t:t[1])[0]
        try:
            clusters[best_mean_key].append(datapoint)
        except KeyError:
            clusters[best_mean_key] = [datapoint]
    return clusters

def reevaluate_cluster_centers(means, clusters):
    # Calculate the new cluster means with the updated cluster assignments
    newmeans = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmeans.append(np.mean(clusters[k], axis = 0))
    return newmeans

def converged(means, oldmeans):
    # Check to see if the new means are the same as the old means
    return (set([tuple(a) for a in means]) == set([tuple(a) for a in oldmeans]))

def k_means_plus_plus(x_train, K):
    # Initialize Means
    oldmeans = k_means_plus_plus_initialize(x_train, K)
    means = k_means_plus_plus_initialize(x_train, K)
    while not converged(means, oldmeans):
        oldmeans = means
        
        # Assign all points to a cluster
        clusters = cluster(x_train, means)
        
        # Re-evaluate cluster centers
        means = reevaluate_cluster_centers(oldmeans, clusters)
        
    cluster_list = []
    for i, j in clusters.iteritems():
        for k in j:
            cluster_list.append([i, k])

    final_array = []
    dummy_array = [0]*len(cluster_list)
    for i in range(len(cluster_list)):
        for j in range(len(x_train)):
            if (x_train[j] == cluster_list[i][1]).all():
                if dummy_array[i] != 1:
                    final_array.append([cluster_list[i][0], j, cluster_list[i][1]])
                dummy_array[i] = 1
                
    return(means, final_array)