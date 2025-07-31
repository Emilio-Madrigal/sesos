import numpy as np

def cargar_datos():
    entradas = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    salidas = np.array([
        [1],
        [0],
        [0],
        [0]
    ])
    return entradas, salidas
