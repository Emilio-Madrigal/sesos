import numpy as np
from . import funciones_activacion as fa

class Neurona:
    def __init__(self,cantidad_entradas):
        self.pesos=np.random.rand(cantidad_entradas)
        self.bias=np.random.rand()
        self.entradas=None
        self.total_entradas=None
        self.salidas=None

    def forward(self,entradas):
        """
        salida = sigmoide(peso ⋅ entrada + bias)
        """
        self.entradas=entradas
        print(f"pesos.shape = {self.pesos.shape}, entradas.shape = {entradas.shape}")
        self.total_entradas=np.dot(self.pesos,entradas)+self.bias
        self.salidas=fa.sigmoide(self.total_entradas)
        return self.salidas

    def backward(self, error_salida, tasa_aprendizaje):

        derivada = fa.derivada_sigmoide(self.total_entrada)
        error_total = error_salida * derivada 
        ajuste_pesos = error_total * self.entradas
        self.pesos -= tasa_aprendizaje * ajuste_pesos
        self.bias -= tasa_aprendizaje * error_total

    def __str__(self):
        return f"Pesos: {self.pesos}, bias: {self.bias}"

