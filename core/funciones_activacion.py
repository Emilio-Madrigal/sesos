import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    s = sigmoide(x)
    return s * (1 - s)
