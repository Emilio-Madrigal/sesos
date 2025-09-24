import tensorflow as tf
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[0],[0],[1]], dtype=np.float32)

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='glorot_uniform')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.fit(X, y, epochs=2000, verbose=0)
print("Predicciones Keras:", model.predict(X).round(3).flatten())
