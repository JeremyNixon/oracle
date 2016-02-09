import math
import random
import copy
from math import e
import pandas as pd
import numpy as np
import sklearn.cross_validation

def init_weight(x, y):
    if x > y:
        temp = x
        x = y
        y = temp
        
    difference = y-x
    to_add = difference * random.random()
    return x + to_add

def hyperbolic_tangent(x):
    return math.tanh(x)

def tanh_derivative(y):
    return 1.0 - y**2

def init_matrix(m, n, content = 0.0):
    matrix = []
    for i in range(m):
        matrix.append([content]*n)
    return matrix

def initialize_network(number_input, number_hidden, number_output, bias_node = 1):
    
    number_input += bias_node
    number_hidden += bias_node
    
    weight_input = init_matrix(number_input, number_hidden)
    weight_output = init_matrix(number_hidden, number_output)
    for i in range(number_input):
        for j in range(number_hidden):
            weight_input[i][j] = init_weight(-1, 1)
    for i in range(number_hidden):
        for j in range(number_output):
            weight_output[i][j] = init_weight(-1, 1)
    
    activation_input = number_input * [1.0]
    activation_hidden = number_hidden * [1.0]
    activation_output = number_output * [1.0]
    
    momentum_input = init_matrix(number_input, number_hidden)
    momentum_output = init_matrix(number_hidden, number_output)
    
    return weight_input, weight_output, momentum_input, momentum_output, activation_input, activation_hidden, activation_output

def forward_pass(inputs, number_input, number_hidden, number_output, weight_input, weight_output, activation_input, activation_hidden, activation_output):
    
    for i in range(number_input):
        activation_input[i] = inputs[i]
    
    for i in range(number_hidden):
        aggregate = 0.0
        for j in range(number_input):
            aggregate += activation_input[j] * weight_input[j][i]
        activation_hidden[i] = hyperbolic_tangent(aggregate)
    
    for i in range(number_output):
        aggregate = 0.0
        for j in range(number_hidden):
            aggregate += activation_hidden[j] * weight_output[j][i]
        activation_output[i] = hyperbolic_tangent(aggregate)
        
    return activation_input, activation_hidden, activation_output

def backpropagation(labels, learning_rate, momentum, number_input, number_hidden, number_output, activation_input, activation_hidden, activation_output, weight_input, weight_output, momentum_input, momentum_output, bias_node = 1):
    
    number_input += bias_node
    number_hidden += bias_node
    
    output_errors = [0.0] * number_output
    for i in range(number_output):
        output_errors[i] = labels[i] - activation_output[i]
        output_errors[i] = tanh_derivative(activation_output[i]) * output_errors[i]
    
    hidden_errors = [0.0] * number_hidden
    for i in range(number_hidden):
        error = 0.0
        for j in range(number_output):
            error += output_errors[j]*weight_output[i][j]
        hidden_errors[i] = tanh_derivative(activation_hidden[i]) * error
        
    for i in range(number_hidden):
        for j in range(number_output):
            delta = output_errors[j]*activation_hidden[i]
            weight_output[i][j] = weight_input[i][j] + learning_rate*delta + momentum*momentum_output[i][j]
            momentum_output[i][j] = delta
            
    for i in range(number_input):
        for j in range(number_hidden):
            delta = hidden_errors[j]*activation_input[i]
            weight_input[i][j] = weight_input[i][j] + learning_rate*delta + momentum*momentum_input[i][j]
            momentum_input[i][j] = delta
    
    sse = 0.0
    for i in range(len(labels)):
        sse += .5 * ((labels[i]-activation_output[i])**2)

    return sse, weight_input, weight_output, momentum_input, momentum_output

def train_classifier(data, number_input, number_hidden, number_output, weight_input, weight_output, momentum_input, momentum_output, activation_input, activation_hidden, activation_output, iterations = 1000, learning_rate = .5, momentum=.1, display = 100):
    for i in xrange(iterations):
        sse = 0.0
        for datapoint in data:
            activation_input, activation_hidden, activation_output = forward_pass(datapoint[0], number_input, number_hidden, number_output, weight_input, weight_output, activation_input, activation_hidden, activation_output)
            to_add, weight_input, weight_output, momentum_input, momentum_output = backpropagation(datapoint[1], learning_rate, momentum, number_input, number_hidden, number_output, activation_input, activation_hidden, activation_output, weight_input, weight_output, momentum_input, momentum_output)
            sse += to_add
            
        if i % display == 0:
            print i,
            print 'error %f' %(sse)
            
    return weight_input, weight_output, activation_input, activation_hidden, activation_output


def test_classifier(x_test, number_input, number_hidden, number_output, weight_input, weight_output, activation_input, activation_hidden, activation_output):
    to_return = []
    for datapoint in x_test:
        ai, ah, ao = forward_pass(datapoint, number_input, number_hidden, number_output, weight_input, weight_output, activation_input, activation_hidden, activation_output)
        to_return.append(copy.copy(ao))
    return to_return

# initialize network variables and arguments, arguments for training, arguments for test,
def nn_classifier(x_train, y_train, x_test, number_hidden, number_output, n_iterations=1000, 
                  learning_rate = 0.1, momentum = 0.1, display = 100):
    
    
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)

    x_train_list = []
    y_train_list = []
    x_test_list = []

    for i in x_train:
        x_train_list.append(list(i))
    for i in y_train:
        y_train_list.append([i])
    for i in x_test:
        x_test_list.append(list(i))

    train = [list(i) for i in zip(x_train_list, y_train_list)]
    test = [list(i) for i in x_test_list]
    
    number_input = len(train[0][0])
    
    weight_input, weight_output, momentum_input, momentum_output, activation_input, activation_hidden, activation_output = initialize_network(number_input, number_hidden, number_output)
    
    weight_input, weight_output, activation_input, activation_hidden, activation_output = train_classifier(train, number_input, number_hidden, number_output, weight_input, weight_output, momentum_input, momentum_output, activation_input, activation_hidden, activation_output, n_iterations, learning_rate, momentum, display)
    predictions = test_classifier(test, number_input, number_hidden, number_output, weight_input, weight_output, activation_input, activation_hidden, activation_output)

    classes = []
    for i in predictions:
        if i[0] > 0.5:
            classes.append(1)
        else:
            classes.append(0)

    return classes