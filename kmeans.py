import numpy as np

def L2_norm(vector):
    count = 0
    for i in vector:
        i = i**2
        count += i
    norm = np.sqrt(count)
    return norm

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

def terminated(means, oldmeans):
    # Check to see if the new means are the same as the old means
    return (set([tuple(a) for a in means]) == set([tuple(a) for a in oldmeans]))

def k_means(x_train, K):
    # Initialize Means
    oldmeans = random.sample(x_train, K)
    means = random.sample(x_train, K)
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
                if dummy_array[j] != 1:
                    final_array.append([cluster_list[i][0], j, cluster_list[i][1]])
                dummy_array[j] = 1
                
    return(final_array)