from core.ann import ANN
from training.training import train
import data.dataset_and as dataset
import numpy as np

# 1. Crear la red neuronal
modelo = ANN([2, 1])  # 2 entradas, 1 capa de 1 neurona

# 2. Entrenar el modelo
train(modelo, dataset.entradas, dataset.salidas, epochs=1000, tasa_aprendizaje=0.1)

# 3. Probar el modelo
print("\n--- Pruebas del modelo entrenado ---")
for entrada, esperado in zip(dataset.entradas, dataset.salidas):
    prediccion = modelo.forward(entrada)
    print(f"Entrada: {entrada}, Esperado: {esperado}, Predicho: {np.round(prediccion, 2)}")
