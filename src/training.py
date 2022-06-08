import tensorflow as tf
import numpy as np
from functools import partial
from src.cumulant_losses import discriminator_cum_loss, generator_cum_loss


@tf.function  # "compile" this function (for graph execution and faster performance)
def train_step_G(y, Z_dim, discriminator, generator, g_optimizer, batch_size, alpha):
    """
        optimize for the Generator NN
    :param y: embedded labels (batch_size, y_dim)
    :param Z_dim: noise dimension
    :param discriminator: discriminator model instance
    :param generator: generator model instance
    :param g_optimizer: generator optimizer instance
    :param batch_size: (int)
    :param alpha: Renyi divergence hyper-parameter (float)
    :return: generator part of the loss after optimization (tensor)
    """

    # sample noise for the Generator input
    Z = sample_Z(batch_size, Z_dim)

    with tf.GradientTape() as generator_tape:
        # build model
        G_sample = generator(tf.concat(axis=1, values=[Z, y]), training=True)
        D_fake = discriminator(tf.concat(axis=1, values=[G_sample, y]), training=False)  # NOTE!! Do not update weights!

        # compute loss
        generator_loss = generator_cum_loss(D_fake, alpha)

    # compute Gradients for generator
    grads_generator_loss = generator_tape.gradient(
        target=generator_loss,
        sources=generator.trainable_variables
    )

    # Apply gradients using optimizer
    g_optimizer.apply_gradients(
        zip(grads_generator_loss, generator.trainable_variables)
    )

    return generator_loss


@tf.function
def train_step_D(x, y, Z_dim, discriminator, generator, d_optimizer, batch_size, alpha):
    """
        optimize for Discriminator without Gradient Penalty
    :param x: training data (batch_size, X_dim)
    :param y: embedded labels (batch_size, y_dim)
    :param Z_dim: noise dimension
    :param discriminator: discriminator model instance
    :param generator: generator model instance
    :param d_optimizer: discriminator optimizer instance
    :param batch_size: (int)
    :param alpha: Renyi divergence hyper-parameter (float)
    :return: discriminator part of the loss after optimization (tensor)
    """
    Z = sample_Z(batch_size, Z_dim)

    with tf.GradientTape() as discriminator_tape:
        # build model
        D_real = discriminator(tf.concat(axis=1, values=[x, y]), training=True)
        G_sample = generator(tf.concat(axis=1, values=[Z, y]), training=False)  # NOTE!! Do not update weights!
        D_fake = discriminator(tf.concat(axis=1, values=[G_sample, y]), training=True)

        # compute loss
        discriminator_loss = discriminator_cum_loss(D_real, D_fake, alpha)

    # compute Gradients for discriminator
    grads_discriminator_loss = discriminator_tape.gradient(
        target=discriminator_loss,
        sources=discriminator.trainable_variables
    )

    # Apply gradients using optimizer
    d_optimizer.apply_gradients(
        zip(grads_discriminator_loss, discriminator.trainable_variables)
    )

    return discriminator_loss


# generate noise
def sample_Z(m, n):
    return np.random.normal(0., 1, size=[m, n])


##########################################
# Gradient Penalty functions
##########################################
@tf.function
def gradient_penalty(discriminator, x, G_sample, y, K, mb_size):
    """
        Gradient Penalty. See paper https://arxiv.org/abs/1704.00028

    :param discriminator: discriminator model instance
    :param x: training data (batch_size, X_dim)
    :param G_sample: generated data
    :param y: embedded labels (batch_size, y_dim)
    :param K: Lipschitz constant
    :param mb_size: (int)
    :return: Gradient Penalty (tensor)
    """
    alpha_gp = tf.random.uniform(shape=[mb_size, 1], minval=0., maxval=1., dtype=tf.float32)
    inter = x + (alpha_gp * (G_sample - x))

    with tf.GradientTape() as tape:
        tape.watch(inter)
        predictions = discriminator(tf.concat(axis=1, values=[inter, y]))

    gradients = tape.gradient(predictions, [inter])[
        0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_pen = tf.reduce_mean(
        tf.math.maximum(tf.zeros([slopes.shape[0]], dtype=tf.float32), (slopes - K)) ** 2)  # two-sided penalty

    gradient_pen = tf.dtypes.cast(gradient_pen, tf.float32)

    return gradient_pen


@tf.function
def train_step_D_GP(x, y, Z_dim, discriminator, generator, d_optimizer, batch_size, lam_gp, K, alpha):
    """
         optimize for Discriminator using Gradient Penalty

    :param x: training data (batch_size, X_dim)
    :param y: embedded labels (batch_size, y_dim)
    :param Z_dim: noise dimension
    :param discriminator: discriminator model instance
    :param generator: generator model instance
    :param d_optimizer: discriminator optimizer instance
    :param batch_size: (int)
    :param lam_gp: Gradient Penalty coefficient (float)
    :param K: Lipschitz constant
    :param alpha: Renyi divergence hyper-parameter (float)
    :return: discriminator part of the loss after optimization (tensor), discriminator loss plus GP loss (tensor)
    """
    Z = sample_Z(batch_size, Z_dim)

    with tf.GradientTape() as discriminator_tape:
        # build model
        D_real = discriminator(tf.concat(axis=1, values=[x, y]), training=True)
        G_sample = generator(tf.concat(axis=1, values=[Z, y]), training=False)  # NOTE!! Do not update weights!
        D_fake = discriminator(tf.concat(axis=1, values=[G_sample, y]), training=True)

        # compute loss
        discriminator_loss = discriminator_cum_loss(D_real, D_fake, alpha)

        # compute GP
        gp = gradient_penalty(partial(discriminator, training=True),
                              x, G_sample, y, K, batch_size)
        total_loss = discriminator_loss + lam_gp * gp

    # compute Gradients for discriminator
    grads_discriminator_loss = discriminator_tape.gradient(
        target=total_loss,
        sources=discriminator.trainable_variables
    )

    # Apply gradients using optimizer
    d_optimizer.apply_gradients(
        zip(grads_discriminator_loss, discriminator.trainable_variables)
    )

    return discriminator_loss, total_loss


@tf.function
def train_step_G_GMM(y, Z_dim, discriminator, generator, g_optimizer, batch_size, alpha):
    """
     optimize for Generator, which is a GMM (or ZIMM)

    :param y: embedded labels (batch_size, y_dim)
    :param Z_dim: noise dimension
    :param discriminator: discriminator model instance
    :param generator: generator model instance
    :param g_optimizer: generator optimizer instance
    :param batch_size: (int)
    :param alpha: Renyi divergence hyper-parameter (float)
    :return: generator part of the loss after optimization (tensor)
    """
    with tf.GradientTape() as generator_tape:
        # build model
        G_sample = generator(y, training=True)
        D_fake = discriminator(tf.concat(axis=1, values=[G_sample, y]), training=False)  # NOTE!! Do not update weights!

        # compute loss
        generator_loss = generator_cum_loss(D_fake, alpha)

    # compute Gradients for generator
    grads_generator_loss = generator_tape.gradient(
        target=generator_loss,
        sources=generator.trainable_variables
    )

    # Apply gradients using optimizer
    g_optimizer.apply_gradients(
        zip(grads_generator_loss, generator.trainable_variables)
    )

    return generator_loss


@tf.function
def train_step_D_GP_GMM(x, y, Z_dim, discriminator, generator, d_optimizer, batch_size, lam_gp, K, alpha):
    """
     optimize for Discriminator, generator is a GMM (or ZIMM), using gradient penalty

    :param x: training data (batch_size, X_dim)
    :param y: embedded labels (batch_size, y_dim)
    :param Z_dim: noise dimension
    :param discriminator: discriminator model instance
    :param generator: generator model instance
    :param d_optimizer: discriminator optimizer instance
    :param batch_size: (int)
    :param lam_gp: Gradient Penalty coefficient (float)
    :param K: Lipschitz constant
    :param alpha: Renyi divergence hyper-parameter (float)
    :return: discriminator part of the loss after optimization (tensor), discriminator loss plus GP loss (tensor)
    """

    with tf.GradientTape() as discriminator_tape:
        # build model
        D_real = discriminator(tf.concat(axis=1, values=[x, y]), training=True)
        G_sample = generator(y, training=False)  # NOTE!! Do not update weights!
        D_fake = discriminator(tf.concat(axis=1, values=[G_sample, y]), training=True)

        # compute loss
        discriminator_loss = discriminator_cum_loss(D_real, D_fake, alpha)

        # compute GP
        gp = gradient_penalty(partial(discriminator, training=True),
                              x, G_sample, y, K, batch_size)
        total_loss = discriminator_loss + lam_gp * gp

        # compute Gradients for discriminator
    grads_discriminator_loss = discriminator_tape.gradient(
        target=total_loss,
        sources=discriminator.trainable_variables
    )

    # Apply gradients using optimizer
    d_optimizer.apply_gradients(
        zip(grads_discriminator_loss, discriminator.trainable_variables)
    )

    return discriminator_loss, total_loss


@tf.function
def train_step_G_GMM_Spen(y, Z_dim, discriminator, generator, g_optimizer, batch_size, spen, alpha):
    """ training of generator, which is a GMM. We are using Sigma Penalty on the Variance matrices of the Gaussians
    :param y: embedded labels (batch_size, y_dim)
    :param Z_dim: noise dimension
    :param discriminator: discriminator model instance
    :param generator: generator model instance
    :param g_optimizer:  generator optimizer instance
    :param batch_size: (int)
    :param spen: sigma penalty coefficient (flaot)
    :param alpha: Renyi divergence hyper-parameter (float)
    :return: generator part of the loss after optimization (tensor)
    """
    with tf.GradientTape() as generator_tape:
        # build model
        G_sample = generator(y, training=True)
        D_fake = discriminator(tf.concat(axis=1, values=[G_sample, y]), training=False)  # NOTE!! Do not update weights!

        # compute loss
        generator_loss = generator_cum_loss(D_fake, alpha)

        # compute Sigma Penalty
        sig_loss = tf.reduce_sum(
            tf.math.reciprocal(  # 1./ X where X is the tensor of the previous step
                tf.reshape(
                    # for flattening tensor of shape (K, X_dim) to (K*X_dim, 1). NOTE [-1] that works recursively!
                    tf.linalg.diag_part(  # for taking the K diagonals of a tensor of shape (K, X_dim, X_dim)
                        tf.matmul(generator.G_S, generator.G_S, transpose_b=True)), [-1]))
        )
        generator_total_loss = generator_loss + spen * sig_loss

    # compute Gradients for generator
    grads_generator_loss = generator_tape.gradient(
        target=generator_total_loss,
        sources=generator.trainable_variables
    )

    # Apply gradients using optimizer
    g_optimizer.apply_gradients(
        zip(grads_generator_loss, generator.trainable_variables)
    )

    return generator_loss, generator_total_loss
