import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    s = sigmoide(x)
    return s * (1 - s)

def relu(x): #Rectified Linear Activation Function
    return np.maximum(0, x)#si es negativo devuelve 0, si es positivo devuelve el mismo valor

def derivada_relu(x):
    return np.where(x > 0, 1, 0)#si es positivo devuelve 1, si es negativo devuelve 0