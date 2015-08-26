from numpy import array, dot, transpose
from numpy.linalg import inv

def linear_regression(x_train, y_train, x_test):
    
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones,X))
    y = np.array(y_train)
    
    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = inv(product)
    w = dot(dot(theInverse, Xt), y)
    
    predictions = []
    x_test = np.array(x_test)
    for i in x_test:
        components = w[1:] * i
        predictions.append(sum(components) + w[0])
        
    return predictions