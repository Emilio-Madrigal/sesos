import tensorflow as tf
from .neurona import Neurona

class Layer:
    def __init__(self, cantidad_entradas, cantidad_neuronas, activacion='sigmoide'):
        self.neuronas = [
            Neurona(cantidad_entradas, activacion)
            for _ in range(cantidad_neuronas)
        ]
        self.cantidad_neuronas = cantidad_neuronas
        self.cantidad_entradas = cantidad_entradas
    
    @tf.function
    def forward(self, entradas):
        salidas = []
        for neurona in self.neuronas:
            salidas.append(neurona.forward(entradas))
        return tf.stack(salidas)
    
    def backward(self, errores_salida, tasa_aprendizaje):
        errores_entrada = tf.zeros_like(self.neuronas[0].ultima_entrada)
        for i, neurona in enumerate(self.neuronas):  # Corregido: era "Neurona"
            error_neurona = neurona.backward(errores_salida[i], tasa_aprendizaje)
            errores_entrada += error_neurona
        
        return errores_entrada
    
    def get_trainable_variables(self):
        variables = []
        for neurona in self.neuronas:
            variables.extend(neurona.get_trainable_variables())
        return variables
    
    def __str__(self):
        return f"Layer({self.cantidad_entradas} -> {self.cantidad_neuronas}):\n" + \
               "\n".join(f"  {i}: {neurona}" for i, neurona in enumerate(self.neuronas))
