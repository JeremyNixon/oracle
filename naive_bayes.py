import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import math

class NaiveBayes():
    def __init__(self):
        self.means = None
        self.variances = None
        self.predictions = None
        self.class_count = None
        self.feature_count = None
        
    def gaussian(self, mean, variance, x):
        normalize = 1 / float(math.sqrt(2 * math.pi * variance))
        exp = math.exp((-1/float(2*variance)) * (x - mean)**2)
        return normalize * exp
    
    def fit(self, x_train, y_train):
        self.class_count = len(np.unique(y_train))
        self.feature_count = x_train.shape[1]
        
        self.means = np.zeros((x_train.shape[1], len(np.unique(y_train))))
        self.variances = np.zeros((x_train.shape[1], len(np.unique(y_train))))
        for i in xrange(x_train.shape[1]):
            for j in xrange(len(np.unique(y_train))):
                self.means[i][j] = np.mean(x_train[y_train==j][:,i])
                self.variances[i][j] = np.var(x_train[y_train==j][:,i])
                
    def predict(self, x_test):
        predictions = np.ones((len(x_test), self.class_count))
        for data_index in xrange(len(x_test)):
            for feature_index in xrange(self.feature_count):
                for class_index in xrange(self.class_count):
                    m = means[feature_index][class_index]
                    v = variances[feature_index][class_index]
                    x = x_test[data_index][feature_index]
                    predictions[data_index][class_index] *= self.gaussian(m, v, x)
        return np.argmax(predictions, axis=1)