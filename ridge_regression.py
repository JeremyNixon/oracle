from numpy import array, dot, transpose
from numpy.linalg import inv
import numpy as np

def ridge_regression(points, lam):
    X = array(points)
    X = array([[1] + list(p[:-1]) for p in X])
    y = array([p[-1] for p in X])
    
    #print X
    #print y
    
    Xt = transpose(X)
    lambda_identity = lam*np.identity(len(Xt))
    theInverse = inv(dot(Xt, X)+lambda_identity)
    w = dot(dot(theInverse, Xt), y)
    return w, lambda x: dot(w,x)