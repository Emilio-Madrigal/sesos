import numpy as np
import data.dataset_and as AND

x = AND.entradas
y = AND.salidas
class train:
    def train(self, X, y, epochs, tasa_aprendizaje):
        self.loss_list = []

        for epoch in range(epochs):
            loss = 0

            for i in range(len(X)):
                salida = self.forward(X[i])
                loss += np.mean((y[i] - salida) ** 2)  # error cuadrático medio (MSE) investigar como se ve matematicamente

                gradiente = 2 * (salida - y[i])  # derivada del MSE respecto a la salida
                self.backward(gradiente, tasa_aprendizaje)  # cambia los pesos

            loss /= len(X)  # Promedio de la pérdida total
            self.loss_list.append(loss)
            print(f"Época {epoch+1}: pérdida = {loss}")
