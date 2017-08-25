import numpy as np

def logistic_regression(x_train, y_train, lr = .01, num_iter=1000, optimizer="sgd", batch_size=32):
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    nrow, ncol = x_train.shape
    n_classes = len(np.unique(y_train))
    
    weights = .01 * np.random.randn(ncol, n_classes)
    for iteration in xrange(num_iter):
        
        if optimizer == "batch":
            # Compute Output
            output = np.exp(np.matmul(x_train, weights))
            softmax = output/np.sum(output, axis=1, keepdims=True)

            # Compute Gradient
            softmax[range(len(y_train)), y_train] -= 1
            grad = np.matmul(x_train.T, softmax/nrow) 

            # Update
            weights -= lr * grad
            
        if optimizer == "sgd":
            # Sample batch from data
            stochastic_sample = np.random.random_integers(0, nrow-1, batch_size)
            
            batch = x_train[stochastic_sample]
            batch_labels = y_train[stochastic_sample]
            
            # Compute Output
            output = np.exp(np.matmul(batch, weights))
            softmax = output/np.sum(output, axis=1, keepdims=True)

            # Compute Gradient
            softmax[range(len(batch_labels)), batch_labels] -= 1
            grad = np.matmul(batch.T, softmax/batch_size) 

            # Update
            weights -= lr * grad
            
    return weights

def evaluate(weights, x_test):
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    output = np.exp(np.matmul(x_test, weights))
    softmax = output/np.sum(output, axis=1, keepdims=True)
    predictions = np.argmax(softmax, axis=1)
    return predictions