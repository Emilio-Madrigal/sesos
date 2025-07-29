import numpy as np
from neurona import Neurona

class Layer:
    def __init__(self, cantidad_entradas, cantidad_neuronas):
        self.neuronas = [Neurona(cantidad_entradas) for _ in range(cantidad_neuronas)]

    def forward(self, entradas):
        """
        Propaga las entradas a través de todas las neuronas del layer.
        """
        return np.array([neurona.forward(entradas) for neurona in self.neuronas])

    def backward(self, errores_salida, tasa_aprendizaje):
        """
        Ajusta los pesos de todas las neuronas en el layer.
        """
        for i, neurona in enumerate(self.neuronas):
            neurona.backward(errores_salida[i], tasa_aprendizaje)

    def __str__(self):
        return "\n".join(str(neurona) for neurona in self.neuronas)