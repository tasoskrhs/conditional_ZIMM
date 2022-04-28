import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np


# Feedforward NN (Functional model type)
def FNN_discriminator(data_dim, y_dim, units_list=[32, 32, 1]):
    # define the model input and first dense module
    inputs = keras.Input(shape=(data_dim + y_dim,))
    dense = layers.Dense(units_list[0], activation="relu", name="layer_0")  # first hidden
    x = dense(inputs)

    # remaining dense layers
    for i, units in enumerate(units_list[1:-1], start=1):
        x = layers.Dense(units, activation="relu", name="layer_%i" % i)(x)

    # output layer
    x = layers.Dense(units_list[-1], name="out_layer")(x)
    outputs = x

    # create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="discriminator")

    return model


def FNN_generator(noise_dim, y_dim, units_list=[32, 32, 1]):
    # define the model input and first dense module
    inputs = keras.Input(shape=(noise_dim + y_dim,))
    dense = layers.Dense(units_list[0], activation="relu", name="layer_0")  # first hidden
    x = dense(inputs)

    # remaining dense layers
    for i, units in enumerate(units_list[1:-1], start=1):
        x = layers.Dense(units, activation="relu", name="layer_%i" % i)(x)

    # output layer
    x = layers.Dense(units_list[-1], name="out_layer")(x)
    # outputs = 50. * tf.nn.tanh(x / 50.)
    outputs = x

    # create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="generator")

    return model


# conditional GMM generator
class GMM_generator(layers.Layer):
    def __init__(self, X_dim, y_dim, mb_size, K):
        super(GMM_generator, self).__init__()
        self.K = K
        self.X_dim = X_dim
        self.e_s = 300  # 200 # 100 # 0.01 #eps in sigmoid steepness, c is shift i.e 1/(1+e^{-e_s(x-c)})

        self.W_gm = tf.Variable(initial_value=tf.zeros(shape=[K, y_dim, X_dim]), trainable=True)
        self.W_gw = tf.Variable(initial_value=tf.zeros(shape=[y_dim, K]), trainable=True)
        self.b_gw = tf.Variable((1 / K) * tf.ones(shape=[K]), trainable=True)
        self.G_S = tf.Variable(np.stack((np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'))
                                        , axis=0)
                               , trainable=True)

        t = (1.5 * 3.14) * (1 + 2 * np.linspace(0, 1 - 1 / K, num=K, dtype='f'))
        q1 = t * np.cos(t)
        q2 = t * np.sin(t)
        Q = np.stack((q1, q2), axis=0)

        self.b_gm = tf.Variable(np.transpose(Q), trainable=True)

    @tf.function
    def call(self, y):
        mb_size = tf.shape(y)[0]
        # Sample u~U[0,1] and z_i ~ {X_dim-dimensional Normal)
        Z = tf.random.normal(shape=(mb_size, self.K, 1, self.X_dim), mean=0., stddev=1)
        u = tf.random.uniform(shape=(mb_size, 1), minval=0, maxval=1.0,
                              dtype=tf.dtypes.float32)

        # weights
        W_lin = tf.matmul(y, self.W_gw)  # shape: [mb_size, K]
        self.G_W = W_lin + self.b_gw  # shape: [mb_size, K]

        # Apply Softmax function on the weights, then Wi's \in [0,1] and I take their cumulants
        W_temp = tf.nn.softmax(self.G_W, axis=1)
        W = tf.cumsum(W_temp, axis=1)  # tf.cumsum([a, b, c])  -> [a, a + b, a + b + c]
        W_pl_0 = tf.concat([tf.zeros([mb_size, 1]), W[:, :-1]], axis=1)

        # characteristics
        U = tf.tile(u, tf.constant([1, self.K]))
        charac = tf.math.sigmoid((U - W_pl_0) * self.e_s)  # W shifted right
        charac -= tf.math.sigmoid((U - W) * self.e_s)

        # reparameterize gaussian (mu + Sigma*N(0,1))
        GAU = tf.matmul(Z, self.G_S)  # size: [mb_size, K, 1, X_dim]
        GAU = tf.reshape(GAU, [mb_size, self.K, self.X_dim])

        # reshape labels y
        y_dim = tf.shape(y)[1] # CHECK!   <------------------------------------
        y = tf.reshape(y, [mb_size, 1, 1, y_dim])
        dum = tf.matmul(y, self.W_gm)  # shape: [mb_size, K, 1, X_dim]
        dum = tf.reshape(dum, [mb_size, self.K, self.X_dim])  # reshape again
        dum = dum + self.b_gm
        GAU = GAU + dum  # include label

        # multiply with the characteristic...
        charac_tr = tf.transpose(charac)  # (mb_size, K) -> (K, mb_size)
        GAU_tr = tf.transpose(GAU)  # (mb_size, K, X_dim) -> (X_dim, K, mb_size)
        tmp_char = tf.multiply(charac_tr, GAU_tr)
        tmp_char = tf.transpose(tmp_char)
        # now reduce accoss K's
        out = tf.reduce_sum(tmp_char, axis=1)

        return out


"""
def GMM_generator(X_dim, y_dim, mb_size, K):
    # tf variables and initialization
    W_gm = tf.Variable(tf.zeros(shape=[K, y_dim, X_dim]))
    W_gw = tf.Variable(tf.zeros(shape=[y_dim, K]))
    b_gw = tf.Variable((1 / K) * tf.ones(shape=[K]))  # equal probability
    G_S = tf.Variable(np.stack((np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'))
                               , axis=0))

    t = (1.5 * 3.14) * (1 + 2 * np.linspace(0, 1 - 1 / K, num=K, dtype='f'))
    q1 = t * np.cos(t)
    q2 = t * np.sin(t)
    Q = np.stack((q1, q2), axis=0)

    b_gm = tf.Variable(np.transpose(Q))

    # Sample u~U[0,1] and z_i ~ {X_dim-dimensional Normal)
    u = tf.random.uniform(shape=(mb_size, 1), minval=0, maxval=1.0,
                          dtype=tf.dtypes.float32)
    Z = tf.random.normal(shape=(mb_size, K, 1, X_dim), mean=0., stddev=1)

    # weights
    W_lin = tf.matmul(y, W_gw)  # shape: [mb_size, K]
    G_W = W_lin + b_gw  # shape: [mb_size, K]

    # Apply Softmax function on the weights, then Wi's \in [0,1] and I take their cumulants
    W_temp = tf.nn.softmax(G_W, axis=1)
    W = tf.cumsum(W_temp, axis=1)  # tf.cumsum([a, b, c])  -> [a, a + b, a + b + c]
    W_pl_0 = tf.concat([tf.zeros([mb_size, 1]), W[:, :-1]], axis=1)

    # characteristics
    e_s = 300  # 200 # 100 # 0.01 #eps in sigmoid steepness, c is shift i.e 1/(1+e^{-e_s(x-c)})
    U = tf.tile(u, tf.constant([1, K]))
    charac = tf.math.sigmoid((U - W_pl_0) * e_s)  # W shifted right
    charac -= tf.math.sigmoid((U - W) * e_s)

    # reparameterize gaussian (mu + Sigma*N(0,1))
    GAU = tf.matmul(Z, G_S)  # size: [mb_size, K, 1, X_dim]
    GAU = tf.reshape(GAU, [mb_size, K, X_dim])

    # reshape labels y
    y = tf.reshape(y, [mb_size, 1, 1, y_dim])  # check if
    dum = tf.matmul(y, W_gm)  # shape: [mb_size, K, 1, X_dim]
    dum = tf.reshape(dum, [mb_size, K, X_dim])  # reshape again
    dum = dum + b_gm
    GAU = GAU + dum  # include label

    # multiply with the characteristic...
    charac_tr = tf.transpose(charac)  # (mb_size, K) -> (K, mb_size)
    GAU_tr = tf.transpose(GAU)  # (mb_size, K, X_dim) -> (X_dim, K, mb_size)
    tmp_char = tf.multiply(charac_tr, GAU_tr)
    tmp_char = tf.transpose(tmp_char)
    # now reduce accoss K's
    out = tf.reduce_sum(tmp_char, axis=1)

    # return out  # size: [mb_size, X_dim]

    # ATTEMPT to create a functional model
    inputs = keras.Input(shape=(y_dim,))
    model = keras.Model(inputs=inputs, outputs=out, name="GMM")

    return model
"""
