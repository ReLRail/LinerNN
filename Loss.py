import numpy as np

def Loss(X,Y):
    return np.sum(np.subtract(X,Y)**2)