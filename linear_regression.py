from numpy import array, dot, transpose
from numpy.linalg import inv

def linear_regression(points):
    X = array(points)
    X = array([[1] + list(p[:-1]) for p in X])
    y = array([p[-1] for p in X])
    
    #print X
    #print y
    
    Xt = transpose(X)
    theInverse = inv(dot(Xt, X))
    w = dot(dot(theInverse, Xt), y)
    return w, lambda x: dot(w,x)