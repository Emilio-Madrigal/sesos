import numpy as np
import funciones_activacion as fa

class Neurona:
    def __init__(self, cantidad_entradas):
        self.pesos = np.random.rand(cantidad_entradas)  # Pesos aleatorios
        self.bias = np.random.rand()  
        self.entradas = None  
        self.salida = None    
        self.total_entrada = None  # Resultado de suma ponderada antes de activación

    def forward(self, entradas):
        """
        Calcula la salida de la neurona usando:
        salida = sigmoide(peso ⋅ entrada + bias)
        """
        self.entradas = entradas
        self.total_entrada = np.dot(self.pesos, entradas) + self.bias
        self.salida = fa.sigmoide(self.total_entrada)
        return self.salida

    def backward(self, error_salida, tasa_aprendizaje):
        """
        Ajusta los pesos y el sesgo usando el error recibido.
        """
        derivada = fa.derivada_sigmoide(self.total_entrada)
        error_total = error_salida * derivada  # δ = error * f'(z)
        
        # Gradiente del peso: δ * entrada
        ajuste_pesos = error_total * self.entradas
        
        # Actualiza los pesos y el sesgo
        self.pesos -= tasa_aprendizaje * ajuste_pesos
        self.bias -= tasa_aprendizaje * error_total

    def __str__(self):
        return f"Pesos: {self.pesos}, bias: {self.bias}"

# Crear una neurona con 2 entradas
neurona = Neurona(cantidad_entradas=2)

# Darle entradas [1.0, 0.0]
salida = neurona.forward(np.array([1.0, 0.0]))
print("Salida inicial:", salida)

# Simular error y entrenar
error = salida - 1.0  # Queríamos que fuera 1, pero no lo fue
neurona.backward(error_salida=error, tasa_aprendizaje=0.1)
print("Después de entrenar:", neurona)
