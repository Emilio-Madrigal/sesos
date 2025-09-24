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
        salida = entradas
        for layer in self.layers:
            salida = layer.forward(salida)
        return salida
    
    def backward(self, loss_gradient, tasa_aprendizaje):
        gradiente = loss_gradient
        for layer in reversed(self.layers):
            gradiente = layer.backward(gradiente, tasa_aprendizaje)
        return gradiente
    
    def get_trainable_variables(self):
        variables = []
        for layer in self.layers:
            variables.extend(layer.get_trainable_variables())
        return variables
    
    def compile_with_optimizer(self, optimizer='adam', learning_rate=0.01):
        if optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate)
        else:
            self.optimizer = optimizer
    
    def train_step_tf(self, X, y):
        with tf.GradientTape() as tape:
            predicciones = self.forward(X)
            loss = tf.reduce_mean(tf.square(y - predicciones))
        
        gradientes = tape.gradient(loss, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(gradientes, self.get_trainable_variables()))
        
        return loss
    
    def __str__(self):
        resultado = "Red Neuronal:\n"
        for i, layer in enumerate(self.layers):
            resultado += f"Capa {i + 1}:\n{layer}\n\n"
        return resultado