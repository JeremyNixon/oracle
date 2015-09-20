import pandas as pd
import numpy as np
import random
import math
import sklearn.cross_validation
import sklearn
from collections import Counter

def separate_by_class(x_train, y_train):
    separated = {}
    for i in range(len(x_train)):
        vector_x = x_train[i]
        y = y_train[i]
        if (y not in separated):
            separated[y] = []
        separated[y].append(vector_x)
    return separated

def summarize(x_train):
    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*x_train)] 
    return summaries

def summarize_by_class(x_train, y_train):
    separated = separate_by_class(x_train, y_train)
    summaries = {}
    for class_value, instances in separated.iteritems():
        summaries[class_value] = summarize(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1/math.sqrt(2*math.pi) * stdev) * exponent

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.iteritems():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] += calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.iteritems():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, x_test):
    predictions = []
    for i in range(len(x_test)):
        result = predict(summaries, x_test[i])
        predictions.append(result)
    return predictions

def naive_bayes(x_train, y_train, x_test):
    summaries = summarize_by_class(x_train, y_train)
    return get_predictions(summaries, x_test)