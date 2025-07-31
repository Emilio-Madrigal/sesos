import numpy as np
from .layer import Layer

class ANN:
    def __init__(self):
        self.layers = []
        self.loss_list = []

    def add_layer(self, cantidad_entradas, cantidad_neuronas):

        if not self.layers:
            self.layers.append(Layer(cantidad_entradas, cantidad_neuronas))
        else:
            cantidad_entradas_anterior=len(self.layer[-1].neuronas)
            self.layers.append(Layer(cantidad_entradas_anterior, cantidad_neuronas))
    def forward(self, entradas):

        for layer in self.layers:
            entradas=layer.forward(entradas)
            return entradas
    
    def backward(self, loss_gradient, tasa_aprendizaje):
        for layer in reversed(self.layers):
            loss_gradient=layer.backward(loss_gradient,tasa_aprendizaje)