import tensorflow as tf


def build_model(layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(layers[0],)))
    for width in layers[1:-1]:
        model.add(tf.keras.layers.Dense(width, activation="tanh", kernel_initializer='glorot_normal'))
    model.add(tf.keras.layers.Dense(layers[-1]))
    return model


# ── FNO building blocks ────────────────────────────────────────────────────────

class SpectralConv1D(tf.keras.layers.Layer):
    """Spectral convolution: multiply the first n_modes Fourier coefficients."""

    def __init__(self, width, n_modes, **kwargs):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        scale = 1.0 / width
        init = tf.keras.initializers.RandomUniform(-scale, scale)
        self.wr = self.add_weight(name="wr", shape=(width, width, n_modes), initializer=init)
        self.wi = self.add_weight(name="wi", shape=(width, width, n_modes), initializer=init)

    def call(self, x):
        # x: (batch, Nx, width)
        Nx = tf.shape(x)[1]
        # FFT along the spatial axis
        x_ft = tf.signal.rfft(tf.transpose(x, [0, 2, 1]))     # (batch, width, Nx//2+1)
        x_modes = x_ft[:, :, :self.n_modes]                   # (batch, width, n_modes)

        w = tf.complex(self.wr, self.wi)                       # (width, width, n_modes)
        out_modes = tf.einsum('biM,ioM->boM', x_modes, w)     # (batch, width, n_modes)

        # Zero-pad back to rfft size and inverse FFT
        n_rfft = Nx // 2 + 1
        pad = tf.zeros([tf.shape(x)[0], tf.shape(w)[1], n_rfft - self.n_modes], dtype=tf.complex64)
        out_ft = tf.concat([out_modes, pad], axis=-1)          # (batch, width, Nx//2+1)
        out = tf.signal.irfft(out_ft)                          # (batch, width, Nx)
        return tf.transpose(out, [0, 2, 1])                    # (batch, Nx, width)


class FNOBlock1D(tf.keras.layers.Layer):
    """One FNO layer: spectral conv + pointwise linear, then GELU."""

    def __init__(self, width, n_modes, **kwargs):
        super().__init__(**kwargs)
        self.spectral = SpectralConv1D(width, n_modes)
        self.W = tf.keras.layers.Dense(width)

    def call(self, x):
        return tf.nn.gelu(self.spectral(x) + self.W(x))


def build_fno_model(nx, nt, width=32, n_modes=16, n_layers=4):
    """FNO that maps (u0(x), x_coords) → u(x, t) on a fixed grid.

    Input shape:  (batch, nx, 2)  — channels are [u0, x]
    Output shape: (batch, nx, nt) — full space-time trajectory
    """
    inputs = tf.keras.Input(shape=(nx, 2))
    x = tf.keras.layers.Dense(width)(inputs)        # lift to FNO width
    for _ in range(n_layers):
        x = FNOBlock1D(width, n_modes)(x)
    x = tf.keras.layers.Dense(128, activation='gelu')(x)
    outputs = tf.keras.layers.Dense(nt)(x)          # project to Nt time steps
    return tf.keras.Model(inputs, outputs)