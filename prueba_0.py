import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsiudos = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#capa=tf.keras.layers.Dense(units=1,input_shape=[1])
#modelo=tf.keras.Sequential([capa])

oculta1=tf.keras.layers.Dense(units=3,input_shape=[1])
oculta2=tf.keras.layers.Dense(units=3)
salida=tf.keras.layers.Dense(units=1)
modelo=tf.keras.Sequential([oculta1,oculta2,salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

historial = modelo.fit(celsiudos, fahrenheit, epochs=600, verbose=False)
print("Modelo entrenado")

print(f"Predicción para 100°C: {modelo.predict(np.array([100.0]), verbose=0)[0][0]:.1f} °F")
print(f"Predicción para 0°C: {modelo.predict(np.array([0.0]), verbose=0)[0][0]:.1f} °F")
print("\n¡Cierra la ventana de la gráfica para terminar el programa!")

plt.xlabel('epochs')
plt.ylabel('magnitud perdida')
plt.plot(historial.history['loss'])
plt.title('Pérdida durante el entrenamiento')
plt.show()