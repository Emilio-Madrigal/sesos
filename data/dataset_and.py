import numpy as np

def cargar_datos():
    entradas = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    salidas = np.array([
        [0],
        [0],
        [0],
        [1]
    ])
    return entradas, salidas
