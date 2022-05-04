""" Losses for Generator and Discriminator"""
import tensorflow as tf


#@tf.function
def discriminator_cum_loss(D_real, D_fake, alpha=0.5):
    beta = 1 - alpha
    gamma = alpha

    # variational representation
    if beta == 0:
        D_loss_real = tf.reduce_mean(D_real)
    else:
        max_val = tf.reduce_max((-beta) * D_real)
        D_loss_real = -(1.0 / beta) * (tf.math.log(tf.reduce_mean(tf.exp((-beta) * D_real - max_val))) + max_val)

    if gamma == 0:
        D_loss_fake = tf.reduce_mean(D_fake)

    else:
        max_val = tf.reduce_max(gamma * D_fake)
        D_loss_fake = (1.0 / gamma) * (tf.math.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)

    D_loss = D_loss_real - D_loss_fake

    return -D_loss # CHECK!!!!!!!!!!!!


#@tf.function
def generator_cum_loss(D_fake, alpha=0.5):
    gamma = alpha

    # variational representation
    if gamma == 0:
        G_loss = -tf.reduce_mean(D_fake)

    else:
        max_val = tf.reduce_max((gamma) * D_fake)
        G_loss = - (1.0 / gamma) * (tf.math.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)

    return G_loss
