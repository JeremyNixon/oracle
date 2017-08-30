import numpy as np

def linear_regression(x_train, y_train, lr=.00001, num_iters=1000):
    # Bias
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    
    # Initialize Weights
    w = .01 * np.ones(x_train.shape[1])
    
    # Gradient Descent
    for iteration in xrange(num_iters):
        # Compute Gradient
        raw_output = np.matmul(x_train, w)
        diff = y_train - raw_output
        grad = np.matmul(x_train.T, diff)
        
        # Take Step in Direction of Gradient
        w += lr * grad
    return w

def predict_lin_reg(weights, x_test):
    # Add Bias
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    
    output = np.matmul(x_test, weights)
    return output