import numpy as np

def logistic_regression(x_train, y_train, lr = .01, num_iter=1000):
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    nrow, ncol = x_train.shape
    n_classes = len(np.unique(y_train))
    
    weights = .01 * np.random.randn(ncol, n_classes)
    for iteration in xrange(num_iter):
        # Forwards
        output = np.exp(np.matmul(x_train, weights))
        softmax = output/np.sum(output, axis=1, keepdims=True)
        
        # Backwards
        softmax[range(len(y_train)), y_train] -= 1
        grad = np.matmul(x_train.T, softmax/nrow) 
        
        # Update
        weights -= lr * grad
    return weights

def evaluate(weights, x_test):
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    output = np.exp(np.matmul(x_test, weights))
    softmax = output/np.sum(output, axis=1, keepdims=True)
    predictions = np.argmax(softmax, axis=1)
    return predictions