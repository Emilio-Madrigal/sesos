import tensorflow as tf
from .layer import Layer

class ANN:
    def __init__(self, estructura, activaciones=None):
        self.layers = []
        self.loss_list = []

        if activaciones is None:
            activaciones = ['sigmoide'] * (len(estructura) - 1)

        for i in range(1, len(estructura)):
            cantidad_entradas = estructura[i - 1]
            cantidad_neuronas = estructura[i]
            activacion = activaciones[i - 1]
            self.layers.append(Layer(cantidad_entradas, cantidad_neuronas, activacion))

    @tf.function
    def forward(self, entradas):
        salida=entradas
        for layer in self.layers:
            salida=layer.forward(salida)
        return salida
    
    def backward(self, loss_gradient, tasa_aprendizaje):
        gradiente=loss_gradient
        for layer in reversed(self.layers):
            gradiente=layer.backward(gradiente,tasa_aprendizaje)
        return gradiente
    
    def get_trainable_variables(self):
        variables = []
        for layer in self.layers:
            variables.extend(layer.get_trainable_variables())
        return variables
    
    def compilar_con_optimizador(self, optimizador='adam', tasa_aprendizaje=0.01):
        if optimizador=='adam':
            self.optimizer = tf.keras.optimizers.Adam(tasa_aprendizaje)
        elif optimizador=='sgd':
            self.optimizador=tf.keras.optimizers.SGD(tasa_aprendizaje)