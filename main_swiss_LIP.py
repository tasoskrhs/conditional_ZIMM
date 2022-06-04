""" Swiss roll data example
    Lipschitz version using gradient penalty and cFNN"""
import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt

from model import FNN_generator, FNN_discriminator
from data_loading import load_data
from training import train_step_D_GP, train_step_G, sample_Z
from cumulant_losses import discriminator_cum_loss, generator_cum_loss
from plotting_functions import plot_color, plot_losses_over_steps


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Swiss Roll Training cFNN'
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
        type=str, default='none'
    )

    return parser.parse_args()



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
    x_data_train, y_data_train, x_data_test, y_data_test = load_data(args.data_fname, NoTest= NoT, missing_labels=args.missing_labels)

    # Set the model up, units_list is the architecture of the layers
    discriminator = FNN_discriminator(data_dim=args.d, y_dim=args.y_dim, units_list=[32, 32, 1])
    print(discriminator.summary())  # Functional model

    generator = FNN_generator(noise_dim=args.Z_dim, y_dim=args.y_dim, units_list=[32, 32, args.d])
    print(generator.summary())

    discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, epsilon=1e-8)
    generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, epsilon=1e-8)

    # Save checkpoints
    if not os.path.exists('./training_checkpoints_swiss_roll_cFNN'):
        os.makedirs('./training_checkpoints_swiss_roll_cFNN')

    checkpoint_dir = './training_checkpoints_swiss_roll_cFNN'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
                                     discriminator_optimizer=discriminator_opt,
                                     generator=generator,
                                     discriminator=discriminator)
    # load checkpoints
    if args.resume_from_iter > 0:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial() #.assert_consumed()
        print("Found checkpoint, resuming from iter: ", args.resume_from_iter)
        # Load previously chosen input data!
        x_data_test = genfromtxt(checkpoint_dir + '/x_data_test.csv', delimiter=',', dtype='float32')
        x_data_train = genfromtxt(checkpoint_dir + '/x_data_train.csv', delimiter=',', dtype='float32')

        NoT = x_data_test.shape[0]
        y_data_test = genfromtxt(checkpoint_dir + '/y_data_test.csv', dtype='float32')
        y_data_test = np.ravel(y_data_test) #np.reshape(y_data_test, (NoT, -1))
        y_data_train = genfromtxt(checkpoint_dir + '/y_data_train.csv', dtype='float32')
        y_data_train = np.ravel(y_data_train) #np.reshape(y_data_train, (y_data_train.shape[0],-1))

    else:
        # save data for further training from checkpoint
        with open(checkpoint_dir + '/x_data_test.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in x_data_test:
                writer.writerow(val)

        with open(checkpoint_dir + '/x_data_train.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in x_data_train:
                writer.writerow(val)

        with open(checkpoint_dir + '/y_data_test.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            y_data_test_ = np.reshape(y_data_test, (NoT, -1))  # undo np.ravel()
            for val in y_data_test_:
                writer.writerow(val)

        with open(checkpoint_dir + '/y_data_train.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            y_data_train_ = np.reshape(y_data_train, (y_data_train.shape[0], -1))  # undo np.ravel()
            for val in y_data_train_:
                writer.writerow(val)

    if not os.path.exists('./output_files'):
        os.makedirs('./output_files')
    if not os.path.exists('./output_files/' + args.output_fname):
        os.makedirs('./output_files/' + args.output_fname)
    if not os.path.exists('./output_files/' + args.output_fname + '/csv'):
        os.makedirs('./output_files/' + args.output_fname + '/csv')

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

    e_1 = np.ones(shape=(1, args.y_dim), dtype=np.float32)
    e_2 = -np.ones(shape=(1, args.y_dim), dtype=np.float32)

    for i in range(args.resume_from_iter, args.steps):

        if i % SF == 0:
            # generate data for plotting
            print('plot generated data')
            y_sample = np.linspace(0., 1.0, num=NoT)
            y_sample_emb = np.transpose(np.tile(np.linspace(0., 1.0, num=NoT), (args.y_dim, 1))) * e_1
            y_sample_emb += np.transpose(np.tile((1 - np.linspace(0., 1.0, num=NoT)), (args.y_dim, 1))) * e_2
            Z = sample_Z(x_data_test.shape[0], args.Z_dim)

            x_gen = generator(tf.concat(axis=1, values=[Z, y_sample_emb]), training=False)

            fig = plot_color(x_gen, y_sample, x_data_test[:NoT, :], 20, 'conditional FNN')
            plt.savefig('output_files/' + args.output_fname + '/plots{}.png'.format(str(i).zfill(3)), bbox_inches='tight',
                        format='png', dpi=100)
            plt.close(fig)

            with open('output_files/' + args.output_fname + '/csv/cFNN_LIP_samples_alpha_' + str(args.alpha)
                      + '_iteration_' + str(i) + '.csv', "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                x_gen_ = x_gen.numpy() # convert tensor to numpy array
                for val in x_gen_:
                    writer.writerow(val)

        for j in range(k):
            # update discriminator
            idx = np.random.randint(x_data_train.shape[0], size=args.mb_size)
            X_mb = x_data_train[idx, :]
            y_mb = y_data_train[idx]

            # label embedding
            y_mb_emb = np.transpose(np.tile(y_mb, (args.y_dim, 1))) * e_1 + \
                       np.transpose(np.tile(1. - y_mb, (args.y_dim, 1))) * e_2

            # discriminator_loss_i = train_step_D(X_mb, y_mb_emb, args.Z_dim, discriminator, generator, discriminator_opt,
            #                                    args.batch_size, args.alpha)
            discriminator_loss_i, total_loss_i = train_step_D_GP(X_mb, y_mb_emb, args.Z_dim, discriminator, generator,
                                                                 discriminator_opt, args.mb_size, args.lam_gp, args.K_lip, args.alpha)

            discriminator_losses.append(discriminator_loss_i)
            total_losses.append(total_loss_i)

        generator_loss_i = train_step_G(y_mb_emb, args.Z_dim, discriminator, generator, generator_opt, args.mb_size, args.alpha)
        generator_losses.append(generator_loss_i)

        if i % steps_per_print == 0:
            # run current model for test sample points
            D_real_tmp = discriminator(tf.concat(axis=1, values=[x_data_test,
                                                                 np.transpose(np.tile(y_data_test, (
                                                                 args.y_dim, 1))) * e_1 + np.transpose(
                                                                     np.tile(1. - y_data_test,
                                                                             (args.y_dim, 1))) * e_2]),
                                       training=False)
            Z = sample_Z(x_data_test.shape[0], args.Z_dim)
            D_fake_tmp = generator(tf.concat(axis=1, values=[Z,
                                                             np.transpose(np.tile(y_data_test, (
                                                             args.y_dim, 1))) * e_1 + np.transpose(
                                                                 np.tile(1. - y_data_test, (args.y_dim, 1))) * e_2]),
                                   training=False)

            generator_val_loss.append(generator_cum_loss(D_fake_tmp, args.alpha))
            discriminator_val_loss.append(discriminator_cum_loss(D_real_tmp, D_fake_tmp, args.alpha))

            print("step: %i, Generator loss: %f, Discriminator loss: %f" % \
                  (i, generator_loss_i, discriminator_loss_i)
                  )

        # Save the model every 100 steps
        if (i + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # plot Losses during steps (over mini-batches)
    plot_losses_over_steps(discriminator_losses, generator_losses, save_fname=args.output_fname, alpha=args.alpha)

    print("end of main!")


if __name__ == "__main__":
    main()
