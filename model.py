import tensorflow as tf
def build_model(layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(layers[0],)))
    for width in layers[1:-1]:
        model.add(tf.keras.layers.Dense(width, activation="tanh", kernel_initializer='glorot_normal'))
    model.add(tf.keras.layers.Dense(layers[-1]))
    return model