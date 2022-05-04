""" Synthetic data example
    Lipschitz version using gradient penalty and ZIMM"""
import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from model import ZIMM_generator, FNN_discriminator
from data_loading import load_synth_data
from training import train_step_D_GP_GMM, train_step_G_GMM, sample_Z
from cumulant_losses import discriminator_cum_loss, generator_cum_loss
from plotting_functions import plot_50genes, plot_losses_over_steps


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Synth mRNA Data Training (ZIMM)'
    )
    # Model to train/test and other parameters
    parser.add_argument(
        '--steps', dest='steps',
        help='number of training steps',
        type=int, default=20000
    )
    parser.add_argument(
        '--d', dest='d',
        help='number of dimensions (input data)',
        type=int, default=50
    )
    parser.add_argument(
        '--mb', dest='mb_size',
        help='mini-batch size',
        type=int, default=1024
    )
    parser.add_argument(
        '--data_fname', dest='data_fname',
        help='directory of input data',
        type=str, default='./input_data/simulated_data/'
    )
    parser.add_argument(
        '--alpha', dest='alpha',
        help='alpha parameter of cumulant loss',
        type=float, default=0.5
    )
    parser.add_argument(
        '--K', dest='K',
        help='number of GMM modes',
        type=int, default=20
    )
    parser.add_argument(
        '--K_lip', dest='K_lip',
        help='Lipschitz constant K',
        type=float, default=1.0
    )
    parser.add_argument(
        '--lam_gp', dest='lam_gp',
        help='gradient penalty coefficient',
        type=float, default=1.0
    )
    parser.add_argument(
        '--Z_dim', dest='Z_dim',
        help='noise dimension (of generator)',
        type=int, default=18
    )
    parser.add_argument(
        '--y_dim', dest='y_dim',
        help='dimension of label embedding',
        type=int, default=10
    )
    parser.add_argument(
        '--spen', dest='spen',
        help='use Sigma Penalty on generator',
        type=float, default=0.001
    )
    parser.add_argument(
        '--lr', dest='lr',
        help='Learning rate for generator and discriminator',
        type=float, default=0.0005
    )
    parser.add_argument(
        '--saved_model_name', dest='saved_model_name',
        help='saved model name',
        type=str, default='ckpt'
    )
    parser.add_argument(
        '--output_fname', dest='output_fname',
        help='name of the output file directory, for this experiment',
        type=str, default='plots_cond_cond_ZIMM_uniform'
    )

    return parser.parse_args()


""" Unit tests """


def test_1():
    assert sum((1, 2, 3)) == 6, "Should be 6"


def main():
    args = parse_args()

    # Check inputs.
    print(args.data_fname)
    if not os.path.exists(args.data_fname):
        raise FileNotFoundError("Input data file does not exist")
    if args.mb_size <= 0:
        raise ValueError("Invalid batch size")
    if args.steps <= 0:
        raise ValueError("Invalid number of steps")

    NoT = 3000  # Number of generated, see later

    # load data
    x_data_train, y_data_train, x_data_test, y_data_test = load_synth_data(args.data_fname, NoTest=NoT)

    # Set the model up
    discriminator = FNN_discriminator(data_dim=args.d, y_dim=args.y_dim, units_list=[32, 32, 1])
    print(discriminator.summary())  # Functional model

    # generator = FNN_generator(noise_dim=args.Z_dim, y_dim=args.y_dim, units_list=[32, 32, args.d])
    # print(generator.summary())
    # generator = GMM_generator(X_dim=args.d, y_dim=args.y_dim, mb_size=args.mb_size, K=args.K) # as a regular function..
    # generator = GMM_generator(X_dim=args.d, y_dim=args.y_dim, mb_size=args.mb_size, K=args.K)  # as a class declaration
    # generator = FNN_generator(noise_dim=args.Z_dim, y_dim=args.y_dim, units_list=[32, 32, args.d])
    generator = ZIMM_generator(X_dim=args.d, y_dim=args.y_dim, K=args.K)  # as a class declaration
    print('instantiated generator...')
    # print(generator(tf.ones((10, args.y_dim))))

    discriminator_opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    generator_opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Save checkpoints
    if not os.path.exists('./training_checkpoints_synth_data_ZIMM'):
        os.makedirs('./training_checkpoints_synth_data_ZIMM')

    checkpoint_dir = './training_checkpoints_synth_data_ZIMM'
    checkpoint_prefix = os.path.join(checkpoint_dir, args.saved_model_name)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
                                     discriminator_optimizer=discriminator_opt,
                                     generator=generator,
                                     discriminator=discriminator)

    if not os.path.exists('./output_files'):
        os.makedirs('./output_files')
    if not os.path.exists('./output_files/' + args.output_fname):
        os.makedirs('./output_files/' + args.output_fname)

    ####################
    #  Training
    ####################

    steps_per_print = 500
    k = 5
    SF = 1000
    batch_size = args.mb_size
    Z_dim = args.Z_dim
    discriminator_losses = []
    generator_losses = []
    total_losses = []

    np.random.seed(0)  # fix seed

    e_1 = np.ones(shape=(1, args.y_dim), dtype=np.float32)
    e_2 = -np.ones(shape=(1, args.y_dim), dtype=np.float32)

    # run training without compiling
    for i in range(args.steps):

        if i % SF == 0:
            # generate data for plotting
            print('plot generated data')
            y_sample = np.linspace(0., 1.0, num=NoT)
            y_sample_emb = np.transpose(np.tile(np.linspace(0., 1.0, num=NoT), (args.y_dim, 1))) * e_1
            y_sample_emb += np.transpose(np.tile((1 - np.linspace(0., 1.0, num=NoT)), (args.y_dim, 1))) * e_2
            # Z = sample_Z(x_data_test.shape[0], Z_dim)  # FNN

            # x_gen = generator(tf.concat(axis=1, values=[Z, y_sample_emb]), training=False)  # FNN
            x_gen = generator(y_sample_emb, training=False)  #GMM

            fig = plot_50genes(samples=x_gen.numpy(), real_samples=x_data_test)
            plt.savefig('output_files/' + args.output_fname + '/plots{}.png'.format(str(i).zfill(3)),
                        bbox_inches='tight',
                        format='png', dpi=100)
            plt.close(fig)

        for j in range(k):
            # update discriminator
            idx = np.random.randint(x_data_train.shape[0], size=batch_size)
            X_mb = x_data_train[idx, :]
            y_mb = y_data_train[idx]

            # embedded labels
            y_mb_emb = np.transpose(np.tile(y_mb, (args.y_dim, 1))) * e_1 + \
                       np.transpose(np.tile(1. - y_mb, (args.y_dim, 1))) * e_2  # CHECK!

            # discriminator_loss_i = train_step_D(X_mb, y_mb_emb, Z_dim, discriminator, generator, discriminator_opt,
            #                                    batch_size)
            # discriminator_loss_i, total_loss_i = train_step_D_GP(X_mb, y_mb_emb, Z_dim, discriminator, generator,
            #                                                     discriminator_opt, batch_size, args.lam_gp, args.K_lip)
            discriminator_loss_i, total_loss_i = train_step_D_GP_GMM(X_mb, y_mb_emb, Z_dim, discriminator, generator,
                                                                     discriminator_opt, batch_size, args.lam_gp,
                                                                     args.K_lip, args.alpha)
            # discriminator_loss_i, total_loss_i = train_step_D_GP(X_mb, y_mb_emb, Z_dim, discriminator, generator,
            #                                                      discriminator_opt, batch_size, args.lam_gp, args.K_lip, args.alpha)

        discriminator_losses.append(discriminator_loss_i)
        total_losses.append(total_loss_i)

        # generator_loss_i = train_step_G(y_mb_emb, Z_dim, discriminator, generator, generator_opt, batch_size)
        # generator_loss_i = train_step_G_GMM(y_mb_emb, Z_dim, discriminator, generator, generator_opt, batch_size)
        # generator_loss_i, _ = train_step_G_GMM_Spen(y_mb_emb, Z_dim, discriminator, generator, generator_opt,
        #                                             batch_size, args.spen, args.alpha)
        # generator_loss_i = train_step_G(y_mb_emb, Z_dim, discriminator, generator, generator_opt, batch_size, args.alpha)
        generator_loss_i = train_step_G_GMM(y_mb_emb, Z_dim, discriminator, generator, generator_opt, batch_size, args.alpha)

        generator_losses.append(generator_loss_i)

        if i % steps_per_print == 0:
            # run current model for test sample points
            D_real_tmp = discriminator(tf.concat(axis=1, values=[x_data_test,
                                                                 np.transpose(np.tile(y_data_test, (
                                                                     args.y_dim, 1))) * e_1 + np.transpose(
                                                                     np.tile(1. - y_data_test,
                                                                             (args.y_dim, 1))) * e_2]),
                                       training=False)

            # Z = sample_Z(x_data_test.shape[0], Z_dim)
            # D_fake_tmp = generator(tf.concat(axis=1, values=[Z,
            #                                                  np.transpose(np.tile(y_data_test, (
            #                                                      args.y_dim, 1))) * e_1 + np.transpose(
            #                                                      np.tile(1. - y_data_test, (args.y_dim, 1))) * e_2]),
            #                        training=False)
            D_fake_tmp = generator(np.transpose(np.tile(y_data_test, (args.y_dim, 1))) * e_1
                                   + np.transpose(np.tile(1. - y_data_test, (args.y_dim, 1))) * e_2,
                                   training=False)

            generator_loss_epoch = generator_cum_loss(D_fake_tmp, args.alpha)
            discriminator_loss_epoch = discriminator_cum_loss(D_real_tmp, D_fake_tmp, args.alpha)

            print("step: %i, Generator loss: %f, Discriminator loss: %f" % \
                  (i, generator_loss_epoch, discriminator_loss_epoch)
                  )

        # Save the model every 1000 steps
        if (i + 1) % 1000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # plot Losses during steps (over mini-batches)
    plot_losses_over_steps(discriminator_losses, generator_losses, save_fname=args.output_fname, alpha=args.alpha)

    print("end of main!")


if __name__ == "__main__":
    test_1()
    print('test_1 passed')
    main()
