"""
    code for paper: GAN-based Training of Semi-Interpretable Generators for Biological Data Interpolation
     https://www.mdpi.com/2076-3417/12/11/5434/htm

    code by: A. Tsourtis (tsourtis@iacm.forth.gr)
"""

""" Swiss roll data example
    Lipschitz version using gradient penalty and cFNN"""
import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
import json

from src.model import FNN_generator, FNN_discriminator
from src.data_loading import load_data
from src.training import train_step_D_GP, train_step_G, sample_Z
from src.cumulant_losses import discriminator_cum_loss, generator_cum_loss
from src.plotting_functions import plot_color, plot_losses_over_steps, plot_test_loss
from src.checkpoint_loading import checkpoint_loading_dataset_handling


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Swiss Roll Training cFNN'
    )
    # Model to train/test and other parameters
    parser.add_argument(
        '--steps', dest='steps',
        help='number of training steps',
        type=int, default=100000
    )
    parser.add_argument(
        '--d', dest='d',
        help='number of dimensions (input data)',
        type=int, default=2
    )
    parser.add_argument(
        '--mb', dest='mb_size',
        help='mini-batch size',
        type=int, default=1024
    )
    parser.add_argument(
        '--data_fname', dest='data_fname',
        help='directory of input data',
        type=str, default='./input_data/swissroll_data/'
    )
    parser.add_argument(
        '--alpha', dest='alpha',
        help='alpha parameter of cumulant loss',
        type=float, default=0.5
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
        type=int, default=2
    )
    parser.add_argument(
        '--y_dim', dest='y_dim',
        help='dimension of label embedding',
        type=int, default=10
    )
    parser.add_argument(
        '--lr', dest='lr',
        help='Learning rate for generator and discriminator',
        type=float, default=0.0002
    )
    parser.add_argument(
        '--saved_model_name', dest='saved_model_name',
        help='saved model name',
        type=str, default='ckpt'
    )
    parser.add_argument(
        '--output_fname', dest='output_fname',
        help='name of the output file directory, for this experiment',
        type=str, default='plots_swissroll_data_cFNN'
    )
    parser.add_argument(
        '--resume_from_iter', dest='resume_from_iter',
        help='steps corresponding to last checkpoint, needed to resume training',
        type=int, default=0
    )
    parser.add_argument(
        '--missing_labels', dest='missing_labels',
        help='Missing labeled data in the training set. Options: none or 0.25_0.3',
        choices=['none', '0.25_0.3'],
        type=str, default='none'
    )
    parser.add_argument(
        '--generate', dest='generate', action='store_true',
        help='Inference only (training should have been already performed!)',
        default=False
    )

    return parser.parse_args()


def generate_data(step, args, x_data_test, NoT, e_1, e_2, generator):
    """
        function used for generating, plotting and saving samples
    :param step: current training step (int)
    :param args: input arguments
    :param x_data_test: test data set
    :param NoT: number of data samples to be generated
    :param e_1: embedding vector (float y_dim-dimensional vector)
    :param e_2: embedding vector (float y_dim-dimensional vector)
    :param generator: instance to the generator network
    :return:
    """

    print('plot generated data')
    y_sample = np.linspace(0., 1.0, num=NoT)
    # label embedding
    y_sample_emb = np.transpose(np.tile(np.linspace(0., 1.0, num=NoT), (args.y_dim, 1))) * e_1
    y_sample_emb += np.transpose(np.tile((1 - np.linspace(0., 1.0, num=NoT)), (args.y_dim, 1))) * e_2

    Z = sample_Z(x_data_test.shape[0], args.Z_dim)

    x_gen = generator(tf.concat(axis=1, values=[Z, y_sample_emb]), training=False)

    fig = plot_color(x_gen, y_sample, x_data_test[:NoT, :], 20, 'conditional FNN')
    plt.savefig('output_files/' + args.output_fname + '/plots{}.png'.format(str(step).zfill(3)), bbox_inches='tight',
                format='png', dpi=100)
    plt.close(fig)

    with open('output_files/' + args.output_fname + '/csv/cFNN_LIP_samples_alpha_' + str(args.alpha)
              + '_iteration_' + str(step) + '.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        x_gen_ = x_gen.numpy()  # convert tensor to numpy array
        for val in x_gen_:
            writer.writerow(val)

    return


def main():
    args = parse_args()

    # Check inputs.
    print(args.data_fname)
    if not os.path.exists(args.data_fname):
        raise FileNotFoundError("Input directory does not exist")
    if args.mb_size <= 0:
        raise ValueError("Invalid batch size")
    if args.steps <= 0:
        raise ValueError("Invalid number of steps")
    if args.steps <= args.resume_from_iter:
        raise ValueError("Invalid iteration to resume from. It should be smaller than STEPS")

    NoT = 500  # Number of generated samples, see later

    # load data
    x_data_train, y_data_train, x_data_test, y_data_test = load_data(args.data_fname, NoTest=NoT,
                                                                     missing_labels=args.missing_labels)

    # Set the model up, units_list is the architecture of the layers
    discriminator = FNN_discriminator(data_dim=args.d, y_dim=args.y_dim, units_list=[32, 32, 1])
    print(discriminator.summary())  # Functional model

    generator = FNN_generator(noise_dim=args.Z_dim, y_dim=args.y_dim, units_list=[32, 32, args.d])
    print(generator.summary())

    discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, epsilon=1e-8)
    generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, epsilon=1e-8)

    # Save checkpoints
    checkpoint_dir = './training_checkpoints_swiss_roll_cFNN'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
                                     discriminator_optimizer=discriminator_opt,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # load checkpoints
    x_data_train, y_data_train, x_data_test, y_data_test = \
        checkpoint_loading_dataset_handling(args, manager, checkpoint, checkpoint_dir,
                                            x_data_train, y_data_train, x_data_test, y_data_test)


    with open('output_files/' + args.output_fname + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # embedding vectors
    e_1 = np.ones(shape=(1, args.y_dim), dtype=np.float32)
    e_2 = -np.ones(shape=(1, args.y_dim), dtype=np.float32)

    ####################
    #  Inference
    ####################
    if args.generate:
        print('generate data in: ' + 'output_files/' + args.output_fname)
        generate_data(0, args, x_data_test, NoT, e_1, e_2, generator)  # choose 0 as the default step id in these files

        return

    ####################
    #  Training
    ####################

    steps_per_print = 5000
    k = 5
    SF = 5000
    discriminator_losses = []
    generator_losses = []
    total_losses = []
    generator_val_loss = []
    discriminator_val_loss = []

    np.random.seed(0)  # fix seed

    for i in range(args.resume_from_iter, args.steps):

        if i % SF == 0:
            # generate data for plotting
            generate_data(i, args, x_data_test, NoT, e_1, e_2, generator)


        for j in range(k):
            # update discriminator
            idx = np.random.randint(x_data_train.shape[0], size=args.mb_size)
            X_mb = x_data_train[idx, :]
            y_mb = y_data_train[idx]

            # label embedding
            y_mb_emb = np.tile(y_mb, (1, args.y_dim)) * e_1 + \
                       np.tile(1. - y_mb, (1, args.y_dim)) * e_2

            # discriminator_loss_i = train_step_D(X_mb, y_mb_emb, args.Z_dim, discriminator, generator, discriminator_opt,
            #                                    args.batch_size, args.alpha)
            discriminator_loss_i, total_loss_i = train_step_D_GP(X_mb, y_mb_emb, args.Z_dim, discriminator, generator,
                                                                 discriminator_opt, args.mb_size, args.lam_gp,
                                                                 args.K_lip, args.alpha)

        discriminator_losses.append(discriminator_loss_i)
        total_losses.append(total_loss_i)

        # update generator
        generator_loss_i = train_step_G(y_mb_emb, args.Z_dim, discriminator, generator, generator_opt, args.mb_size,
                                        args.alpha)
        generator_losses.append(generator_loss_i)

        if i % steps_per_print == 0:
            # run current model for test sample points
            D_real_tmp = discriminator(tf.concat(axis=1, values=[x_data_test,
                                                                 np.tile(y_data_test, (1, args.y_dim)) * e_1 +
                                                                 np.tile(1. - y_data_test, (1, args.y_dim)) * e_2]),
                                       training=False)
            Z = sample_Z(x_data_test.shape[0], args.Z_dim)
            D_fake_tmp = generator(tf.concat(axis=1, values=[Z,
                                                             np.tile(y_data_test, (1, args.y_dim)) * e_1 +
                                                             np.tile(1. - y_data_test, (1, args.y_dim)) * e_2]),
                                   training=False)

            generator_val_loss.append(generator_cum_loss(D_fake_tmp, args.alpha))
            discriminator_val_loss.append(discriminator_cum_loss(D_real_tmp, D_fake_tmp, args.alpha))

            print("step: %i, Generator loss: %f, Discriminator loss: %f" % \
                  (i, generator_loss_i, discriminator_loss_i)
                  )

        # Save the model every 100 steps
        if (i + 1) % 5000 == 0:
            manager.save()

    # plot training and test Losses during steps (over mini-batches)
    plot_losses_over_steps(discriminator_losses, generator_losses, save_fname=args.output_fname, alpha=args.alpha)
    plot_test_loss(generator_val_loss, discriminator_val_loss, save_fname=args.output_fname, alpha=args.alpha)

    print("end of main!")


if __name__ == "__main__":
    main()
