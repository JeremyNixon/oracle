# Generalization to Numpy made by Jeremy Nixon, credits to James Howard and Neil Schemenauer for the pure python implementation.

import math
import random
from math import e
import pandas as pd
import numpy as np
import sklearn.cross_validation

random.seed(0)

def rand(a, b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no, regression = False):

        self.regression = regression

        #Number of input, hidden and output nodes.
        self.ni = ni  + 1 # +1 for bias node
        self.nh = nh  + 1 # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-1, 1)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-1, 1)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)


    def update(self, inputs):
        if len(inputs) != self.ni-1:
            print len(inputs)
            print self.ni-1
            raise ValueError, 'wrong number of inputs'

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            total = 0.0
            for i in range(self.ni):
                total += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(total)

        # output activations
        for k in range(self.no):
            total = 0.0
            for j in range(self.nh):
                total += self.ah[j] * self.wo[j][k]
            self.ao[k] = total
            if not self.regression:
                self.ao[k] = sigmoid(total)
        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            output_deltas[k] = targets[k] - self.ao[k]
            if not self.regression:
                output_deltas[k] = dsigmoid(self.ao[k]) * output_deltas[k]


        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5*((targets[k]-self.ao[k])**2)
        return error


    def test(self, patterns, verbose = False):
        tmp = []
        for p in patterns:
            if verbose:
                print p, '->', self.update(p)
            tmp.append(self.update(p))
        return tmp


    def weights(self):
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]

    def train(self, patterns, iterations=1000, N=0.5, M=0.1, display = 100, verbose = False):
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                self.update(p[0])
                tmp = self.backPropagate(p[1], N, M)
                error += tmp

            if i % display == 0:
                print i,
                print 'error %-14f' % error

def nn_classifier(x_train, y_train, x_test, n_hidden = 2, n_output = 1, n_iterations = 1000, lr = 0.5, momentum = 0.1, display = 100):
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

    n = NN(len(train[0][0]), n_hidden, n_output, regression = False)

    n.train(train, n_iterations, lr, momentum, display)
    predictions = n.test(test, verbose = True)

    classes = []
    for i in predictions:
        if i[0] > 0.5:
            classes.append(1)
        else:
            classes.append(0)

    return classes