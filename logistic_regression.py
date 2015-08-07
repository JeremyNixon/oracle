from __future__ import division
import numpy as np
import scipy
import math
from scipy.optimize import fmin_bfgs

def train(data, Olabels):
    unique_classes = np.unique(Olabels)
    m,n = data.shape
    
    labels = np.zeros(Olabels.shape)
    
    uniq_Olabel_names = np.unique(Olabels)
    uniq_label_list = range(len(uniq_Olabel_names))
    
    for each in zip(uniq_Olabel_names, uniq_label_list):
        o_label_name = each[0]
        new_label_name = each[1]
        labels[np.where(Olabels == o_label_name)] = new_label_name
    
    labels = labels.reshape((len(labels),1))
    num_classes = len(unique_classes)
    
    Init_Thetas = []
    Thetas = []
    Cost_Thetas = []
    Cost_History_Theata = []
    
    if(num_classes == 2):
        theta_init = np.zeros((n,1))
        Init_Thetas.append(theta_init)
        local_labels = labels
        init_theta = Init_Thetas[0]
        new_theta, final_cost = computeGradient(data, local_labels, init_theta)
        
        Thetas.append(new_theta)
        Cost_Thetas.append(final_cost)
        
    elif(num_classes > 2):
        for eachInitTheta in range(num_classes):
            theta_init = np.zeros((n,1))
            Init_Thetas.append(theta_init)
            pass
        
        for eachClass in range(num_classes):
            local_labels = np.zeros(labels.shape)
            local_labels[np.where(labels == eachClass)] = 1
            init_theta = Init_Thetas[eachClass]
            
            new_theta, final_cost = computeGradient(data, local_labels, init_theta)
            Thetas.append(new_theta)
            Cost_Thetas.append(final_cost)
    return Thetas, Cost_Thetas

def computeGradient(data, labels, init_theta):
    alpha = 1
    num_iters = 100
    m,n = data.shape
    regularized = False
    
    if(regularized == True):
        llambda = 1
    else:
        llambda = 0
    
    for eachIteration in range(num_iters):
        cost = computeCost(data, labels, init_theta)
        B = sigmoidCalc(np.dot(data, init_theta) - labels)
        A = (1/m)*np.transpose(data)
        grad = np.dot(A,B)
        
        A = sigmoidCalc(np.dot(data, init_theta)) - labels
        B = data[:,0].reshape((data.shape[0],1))
        grad[0] = (1/m) * np.sum(A*B)
        
        A = sigmoidCalc(np.dot(data, init_theta)) - labels
        B = data[:,range(1,n)]
        
        for i in range(1, len(grad)):
            A = sigmoidCalc(np.dot(data, init_theta)) - labels
            B = data[:,i].reshape((data[:,i].shape[0],1))
            grad[i] = (1/m)*np.sum(A*B) + ((llambda/m)*init_theta[i])
        
        init_theta = init_theta - (np.dot((alpha/m), grad))
        
    return init_theta, cost

def sigmoidCalc(data):
    data = np.array(data, dtype = np.longdouble)
    g = 1/(1+np.exp(-data))
    return g

def computeCost(data, labels, init_theta):
    regularized = False
    if regularized == True:
        llambda = 1.0
    else:
        llambda = 0
    
    m,n = data.shape
    
    J = 0
    
    grad = np.zeros(init_theta.shape)
    
    theta2 = init_theta[range(1, init_theta.shape[0]),:]
    regularized_parameter = np.dot(llambda/(2*m), np.sum(theta2 * theta2))
    
    J = (-1.0/m) * (np.sum(np.log(sigmoidCalc(np.dot(data, init_theta))) * labels + (np.log(1 - sigmoidCalc(np.dot(data, init_theta))) * (1-labels))))
    J = J + regularized_parameter
    
    return J

def classify(data, Thetas):
    if(len(Thetas) > 1):
        mvals = []
        for eachTheta in Thetas:
            mvals.append(sigmoidCalc(np.dot(data, eachTheta)))
            pass
        return mvals.index(max(mvals))+1

    elif(len(Thetas == 1)):
        cval = round(sigmoidCalc(np.dot(data, Thetas[0])))+1.0
        return cval

def logistic_regression(x_train, y_train, x_test):
    thetas, cost_thetas = train(x_train, y_train)
    labels = []
    for i in x_test:
        labels.append(classify(i, thetas))
    return np.array(labels)