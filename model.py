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


# variant of FNN using gating on the output layer
def FNN_Gated_generator(noise_dim, y_dim, units_list=[32, 32, 1]):
    # define the model input and first dense module
    inputs = keras.Input(shape=(noise_dim + y_dim,))
    dense = layers.Dense(units_list[0], activation="relu", name="layer_0")  # first hidden
    x = dense(inputs)

    # remaining dense layers
    #for i, units in enumerate(units_list[1:-2], start=1):
    #    x = layers.Dense(units, activation="relu", name="layer_%i" % i)(x)
    x_relu = layers.Dense(units_list[1], activation="relu", name="layer_1")(x)

    # Gating layer
    x_g = layers.Dense(units_list[1], activation="sigmoid", name="gated_layer")(x)

    # output layer
    x = layers.Dense(units_list[-1], name="out_layer")(x_relu * x_g)

    outputs = x

    # create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="Gated_generator")

    return model

# conditional GMM generator
class GMM_generator(layers.Layer):
    def __init__(self, X_dim, y_dim, K):
        super(GMM_generator, self).__init__()
        self.K = K
        self.X_dim = X_dim
        self.e_s = 300  # 200 # 100 # 0.01 #eps in sigmoid steepness, c is shift i.e 1/(1+e^{-e_s(x-c)})

        # initialize for faster training on SwissRoll
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


# conditional ZIMM generator
class ZIMM_generator(layers.Layer):
    def __init__(self, X_dim, y_dim, K):
        super(ZIMM_generator, self).__init__()
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
                                         np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f')
                                         # np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         # np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         # np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         # np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f'),
                                         # np.identity(X_dim, dtype='f'), np.identity(X_dim, dtype='f')
                                         ),
                                        axis=0)
                               , trainable=True)

        # initialize for faster training using K-means


        self.b_gm = tf.Variable((tf.random.normal(shape=[K, X_dim], mean=0., stddev=1)))
        self.G_a = tf.Variable(tf.random.uniform(shape=[K, X_dim], minval=0.0, maxval=1.0))

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

        # 2nd Characteristic for Bernoulli-Gaussian mixture, that will give weights a_k and a_k-1 respectively
        u_2 = tf.random.uniform(shape=[mb_size, 1, self.X_dim], minval=0, maxval=1.0)
        U_2 = tf.tile(u_2, tf.constant([1, self.K, 1]))
        #one Characteristic per K per X_dim
        charac_2 = tf.math.sigmoid(U_2 * self.e_s)
        charac_2 -= tf.math.sigmoid( (U_2 - self.G_a) * self.e_s)

        # reparameterize gaussian (mu + Sigma*N(0,1))
        GAU = tf.matmul(Z, self.G_S)  # size: [mb_size, K, 1, X_dim]
        GAU = tf.reshape(GAU, [mb_size, self.K, self.X_dim])

        # reshape labels y
        y_dim = tf.shape(y)[1]
        y = tf.reshape(y, [mb_size, 1, 1, y_dim])
        dum = tf.matmul(y, self.W_gm)  # shape: [mb_size, K, 1, X_dim]
        dum = tf.reshape(dum, [mb_size, self.K, self.X_dim])  # reshape again
        dum = dum + self.b_gm
        GAU = GAU + dum  # include label

        # First multiply Bernoulli and Gaussian with their weights a_k and 1-a_k respectively
        Ber = tf.zeros([mb_size, self.K, self.X_dim])
        tmp_char2_Gau = tf.multiply(charac_2, GAU)

        # then add them for Gaussian-Bernoulli Mixture
        GBM = Ber + tmp_char2_Gau
        GBM_tr = tf.transpose(GBM)

        # multiply with the characteristic...
        charac_tr = tf.transpose(charac)  # (mb_size, K) -> (K, mb_size)
        tmp_char = tf.multiply(charac_tr, GBM_tr)
        tmp_char = tf.transpose(tmp_char)
        # now reduce accoss K's
        out = tf.reduce_sum(tmp_char, axis=1)

        return out

