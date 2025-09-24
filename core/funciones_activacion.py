import tensorflow as tf

def sigmoide(x):
    return tf.nn.sigmoid(x)

def derivada_sigmoide(x):
    s=sigmoide(x)
    return s*(1-s)

def relu(x):
    return tf.nn.relu(x)

def derivada_relu(x):
    return tf.cast(x>0,tf.float32)

def tanh(x):
    return tf.nn.tanh(x)

def derivada_tanh(x):
    return 1-tf.square(tf.nn.tanh(x))