import numpy as np

def lasso(x_train, y_train, lr=.01, num_iter=10000, lam = 1):
    # Add Bias
    ones = np.ones(len(x_train))
    x_train = np.column_stack((ones, x_train))
    
    # Get number of rows and columns in feature matrix
    nrow, ncol = x_train.shape

    weights = np.zeros(ncol)
    for iteration in xrange(num_iter):
        # Compute errors
        error = y_train - np.matmul(x_train, weights)

        # Compute gradient
        l1 = lam * np.sign(weights)
        grad = sum([error[i] * x_train[i] for i in xrange(nrow)])/nrow - l1
        
        # Update weights
        weights = weights + lr * grad
    return weights