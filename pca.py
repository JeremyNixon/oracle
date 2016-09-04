import numpy as np

def PCA(X):
    covariance = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    projection = np.dot(eigenvectors.T, X.T)
    return projection, eigenvalues, eigenvectors