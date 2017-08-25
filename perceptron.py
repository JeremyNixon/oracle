import numpy as np

def perceptron(x_train, y_train, lr=.01, num_iters=1000):
    # Add bias
    x_train = np.column_stack((np.ones(x_train.shape[0]), x_train))
    
    # Initialize weights
    weights = np.ones(x_train.shape[1]) * .01
    
    for iteration in xrange(num_iters):
        output = np.round(np.matmul(x_train, weights))
        gradient = np.matmul(y_train-output, x_train)/len(x_train)
        
        weights += lr * gradient
    return weights

def perceptron_evaluate(weights, x_test):
    x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))
    output = np.round(np.matmul(x_test, weights))
    return [int(round(i)) for i in output]