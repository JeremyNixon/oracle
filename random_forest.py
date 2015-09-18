import math
import random
import pandas as pd
import numpy as np
import sklearn.cross_validation

class Tree(object):
    def __init__(self, parents=None):
        self.children = []
        self.split_feature = None
        self.split_feature_value = None
        self.parents = parents
        self.label = None
        
def data_to_distribution(y_train):
        types = set(y_train)
        distribution = []
        for i in types:
            distribution.append(list(y_train).count(i)/float(len(y_train)))
        return distribution
    
def entropy(distribution):
    return -sum([p * math.log(p,2) for p in distribution])

def split_data(x_train, y_train, feature_index):
    attribute_values = x_train[:,feature_index]
    for attribute in set(attribute_values):
        data_subset = []
        for index, point in enumerate(x_train):
            if point[feature_index] == attribute:
                data_subset.append([point, y_train[index]])
        yield data_subset
        
def gain(x_train, y_train, feature_index):
    entropy_gain = entropy(data_to_distribution(y_train))
    for data_subset in split_data(x_train, y_train, feature_index):
        entropy_gain -= entropy(data_to_distribution([label
                    for (point, label) in data_subset]))
    return entropy_gain

def homogeneous(y_train):
    return len(set(y_train)) <= 1

def majority_vote(y_train, node):
    labels = y_train
    choice = max(set(labels), key=list(labels).count)
    node.label = choice
    return node

def build_decision_tree(x_train, y_train, root, remaining_features):
    remaining_features = np.array(list(remaining_features))
    if homogeneous(y_train):
        root.label = y_train[0]
        return root
    
    if remaining_features.shape == 0:
        return majority_vote(y_train, root)
    
    indices = np.random.choice(int(remaining_features.shape[0]), int(2*remaining_features.shape[0]/3), replace = False)

    best_feature = max(remaining_features[indices], key=lambda index: 
                       gain(x_train, y_train, index))
    remaining_features = set(remaining_features)
    if gain(x_train, y_train, best_feature) == 0:
        return majority_vote(y_train, root)
    
    root.split_feature = best_feature
    
    for data_subset in split_data(x_train, y_train, best_feature):
        child = Tree(parents = root)
        child.split_feature_value = data_subset[0][0][best_feature]
        root.children.append(child)
        
        new_x = np.array([point for (point, label) in data_subset])
        new_y = np.array([label for (point, label) in data_subset])
        
        build_decision_tree(new_x, new_y, child, remaining_features - set([best_feature]))
    
    return root

def decision_tree(x_train, y_train):
    return build_decision_tree(x_train, y_train, Tree(), 
                               set(range(len(x_train[0]))))
def find_nearest(array, value):
    nearest = (np.abs(array-value)).argmin()
    return array[nearest]

def classify(tree, point):
    if tree.children == []:
#         print "label = %r" %(tree.label)
        return tree.label
    else:
        try:
            matching_children = [child for child in tree.children
                if child.split_feature_value == point[tree.split_feature]]
            return classify(matching_children[0], point)
        except:
            array = [child.split_feature_value for child in tree.children]
            point[tree.split_feature] = find_nearest(array, point[tree.split_feature])
            matching_children = [child for child in tree.children
                if child.split_feature_value == point[tree.split_feature]]
            return classify(matching_children[0], point)
        
def text_classification(x_test, tree):
    predicted_labels = [classify(tree, point) for point in x_test]
    return predicted_labels

def random_forest(x_train, y_train, x_test, n_estimators = 100):


    x_train_copy = x_train
    y_train_copy = y_train
    x_test_copy = x_test
    labels = []
    sample = []
    predictions = []
    for i in range(n_estimators):
        sample.append(np.random.choice(len(x_train), len(x_train),
                                       replace=True))
    for i in range(n_estimators):
        x_train = x_train_copy.copy()
        y_train = y_train_copy.copy()
        x_test_copy = x_test_copy.copy()
        
        x = x_train[sample[i]]
        y = y_train[sample[i]]
        tree = decision_tree(x, y)
        labels.append(text_classification(x_test_copy, tree))
    
    
    for index in range(len(labels[0])):
        prediction_dictionary = {}
        for tree_result in range(len(labels)):
            try:
                prediction_dictionary[labels[tree_result][index]] += 1
            except KeyError:
                prediction_dictionary[labels[tree_result][index]] = 1
        store = 0
        for index, value in prediction_dictionary.iteritems():
            if value > store:
                store = value
                chosen = index
        predictions.append(chosen)
        

        
    return predictions


    