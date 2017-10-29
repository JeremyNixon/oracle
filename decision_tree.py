import math
import pandas as pd
import numpy as np


class TreeNode(object):
    def __init__(self, parents=None):
        self.children = []
        self.split_feature = None
        self.split_feature_value = None
        self.parents = parents
        self.label = None

        
class DecisionTreeClassifier():
    def __init__(self):
        self.fit_tree = None
    
    def data_to_distribution(self, y_train):
            types = set(y_train)
            distribution = []
            for i in types:
                distribution.append(list(y_train).count(i)/float(len(y_train)))
            return distribution


    def entropy(self, distribution):
        return -sum([p * math.log(p,2) for p in distribution])


    def split_data(self, x_train, y_train, feature_index):
        attribute_values = x_train[:,feature_index]
        for attribute in set(attribute_values):
            data_subset = []
            for index, point in enumerate(x_train):
                if point[feature_index] == attribute:
                    data_subset.append([point, y_train[index]])
            yield data_subset



    def gain(self, x_train, y_train, feature_index):
        entropy_gain = self.entropy(self.data_to_distribution(y_train))
        for data_subset in self.split_data(x_train, y_train, feature_index):
            entropy_gain -= \
                self.entropy(self.data_to_distribution([label for (point, label) in data_subset]))
        return entropy_gain


    def homogeneous(self, y_train):
        return len(set(y_train)) <= 1

    def majority_vote(self, y_train, node):
        labels = y_train
        choice = max(set(labels), key=list(labels).count)
        node.label = choice
        return node


    def build_decision_tree(self, x_train, y_train, root, remaining_features):
        if self.homogeneous(y_train):
            root.label = y_train[0]
            return root

        if len(remaining_features) == 0:
            return self.majority_vote(y_train, root)

        best_feature = max(remaining_features, key=lambda index: 
                           self.gain(x_train, y_train, index))

        if self.gain(x_train, y_train, best_feature) == 0:
            return self.majority_vote(y_train, root)

        root.split_feature = best_feature

        for data_subset in self.split_data(x_train, y_train, best_feature):
            child = TreeNode(parents = root)
            child.split_feature_value = data_subset[0][0][best_feature]
            root.children.append(child)

            new_x = np.array([point for (point, label) in data_subset])
            new_y = np.array([label for (point, label) in data_subset])

            self.build_decision_tree(new_x, new_y, child, remaining_features - set([best_feature]))

        return root
    
    def find_nearest(self, array, value):
        nearest = (np.abs(array-value)).argmin()
        return array[nearest]


    def classify(self, tree, point):
        if tree.children == []:
            return tree.label
        else:
            try:
                matching_children = [child for child in tree.children
                    if child.split_feature_value == point[tree.split_feature]]
                return self.classify(matching_children[0], point)
            except:
                array = [child.split_feature_value for child in tree.children]
                point[tree.split_feature] = self.find_nearest(array, point[tree.split_feature])
                matching_children = [child for child in tree.children
                    if child.split_feature_value == point[tree.split_feature]]
                return self.classify(matching_children[0], point)


    def text_classification(self, x_train, tree):
        predicted_labels = [self.classify(tree, point) for point in x_train]
        return predicted_labels

    def fit(self, x_train, y_train):
        tree = self.build_decision_tree(x_train, y_train, TreeNode(), 
                                   set(range(len(x_train[0]))))
        self.fit_tree = tree
        
    def predict(self, x_test):
        predictions = self.text_classification(x_test, self.fit_tree)
        return predictions

iris = pd.read_csv('/Users/jeremynixon/Dropbox/python_new/oracle/data/iris_2_classes.csv')
labels = np.array(iris['label'])
features = np.array(iris.drop(['label'], 1))

import sklearn.model_selection
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size = .20, random_state=42)


dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
preds = dt.predict(x_test)

from collections import Counter
accuracy = Counter(preds-y_test)[0]/float(len(y_test))
print "Accuracy = %r" %(accuracy)