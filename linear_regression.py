import numpy as np

def linear_regression(x_train, y_train, lr=.0001, num_iters=1000, optimizer="sgd"):
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    nrow, ncol = x_train.shape
    
    if optimizer == "normal":
        pseudoinverse = np.linalg.inv(np.matmul(x_train.T, x_train))
        weights = np.matmul(pseudoinverse, np.matmul(x_train.T, y_train))
        return weights
    
    weights = .01 * np.random.randn(ncol)
    for iteration in xrange(num_iters):
        
        if optimizer == "sgd":    
            stochastic_sample = np.random.random_integers(0, nrow-1, nrow)
            for index in stochastic_sample:
                weights += lr * (y_train[index]-np.dot(x_train[index], weights)) * x_train[index]

        if optimizer == "batch":
            weights += lr * np.matmul((y_train - np.matmul(x_train, weights)), x_train)/float(len(x_train))

    return weights

def evaluate(x_test, weights):
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    return np.matmul(x_test, weights)