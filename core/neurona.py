import tensorflow as tf
from . import funciones_activacion as fa

class Neurona:
    def __init__(self, cantidad_entradas, activacion):
        self.pesos = tf.Variable(
            tf.random.normal([cantidad_entradas]),
            trainable=True,
            name="pesos"
        )
        
        self.bias = tf.Variable(
            tf.random.normal([1]),
            trainable=True,
            name="bias"
        )
        
        self.activacion_nombre = activacion
        if activacion == 'sigmoide':
            self.activacion = fa.sigmoide
            self.derivada = fa.derivada_sigmoide
        elif activacion == 'relu':
            self.activacion = fa.relu 
            self.derivada = fa.derivada_relu
        elif activacion == 'tanh':
            self.activacion = fa.tanh
            self.derivada = fa.derivada_tanh
        
        self.ultima_entrada = None
        self.ultima_salida_lineal = None
        self.ultima_salida = None
    
    @tf.function
    def forward(self, entradas):
        self.ultima_entrada = entradas

        self.ultima_salida_lineal = tf.reduce_sum(self.pesos * entradas) + self.bias[0]
        
        self.ultima_salida = self.activacion(self.ultima_salida_lineal)
        return self.ultima_salida
    
    def backward(self, error_salida, tasa_aprendizaje):
        with tf.GradientTape() as tape:
            salida_lineal = tf.reduce_sum(self.pesos * self.ultima_entrada) + self.bias[0]
            salida = self.activacion(salida_lineal)
            
            loss = tf.square(error_salida)

        gradientes = tape.gradient(loss, [self.pesos, self.bias])
        
        self.pesos.assign_sub(tasa_aprendizaje * gradientes[0])
        self.bias.assign_sub(tasa_aprendizaje * gradientes[1])
        
        return tf.reduce_sum(gradientes[0])
    
    def get_trainable_variables(self):
        return [self.pesos, self.bias]
    
    def __str__(self):
        return f"Neurona({self.activacion_nombre}): Pesos={self.pesos.numpy()}, Bias={self.bias.numpy()}"
