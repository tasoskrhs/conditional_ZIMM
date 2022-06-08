import csv
from numpy import genfromtxt
import tensorflow as tf
import numpy as np
import os

def checkpoint_loading_dataset_handling(args, manager, checkpoint, checkpoint_dir, x_data_train, y_data_train, x_data_test, y_data_test):
    """
        load checkpoints and update train/test

    :param args: input arguments
    :param manager: tensorflow checkpoint manager (tf.train.CheckpointManager)
    :param checkpoint: tensorflow checkpoint
    :param checkpoint_dir: checkpoints directory
    :param x_data_train: training data set
    :param y_data_train: labels of training data set
    :param x_data_test: test data set
    :param y_data_test: labels of test data set
    :return: old train-test data sets (if resumed from checkpoint)
    """

    if args.resume_from_iter > 0 or args.generate:
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))

        #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial() #.assert_consumed()
        checkpoint.restore(manager.latest_checkpoint)
        print("Found checkpoint, resuming from iter: ", args.resume_from_iter)
        # Load previously chosen input data!
        x_data_test = genfromtxt(checkpoint_dir + '/x_data_test.csv', delimiter=',', dtype='float32')
        x_data_train = genfromtxt(checkpoint_dir + '/x_data_train.csv', delimiter=',', dtype='float32')

        # NoT = x_data_test.shape[0]
        y_data_test = genfromtxt(checkpoint_dir + '/y_data_test.csv', dtype='float32')
        y_data_test = np.reshape(y_data_test, (y_data_test.shape[0], -1)) # np.ravel(y_data_test) #np.reshape(y_data_test, (NoT, -1))
        y_data_train = genfromtxt(checkpoint_dir + '/y_data_train.csv', dtype='float32')
        y_data_train = np.reshape(y_data_train, (y_data_train.shape[0],-1)) # np.ravel(y_data_train) #

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
            y_data_test_ = y_data_test # np.ravel(y_data_test) # np.reshape(y_data_test, (y_data_test.shape[0], -1))  #
            for val in y_data_test_:
                writer.writerow(val)

        with open(checkpoint_dir + '/y_data_train.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            y_data_train_ = y_data_train # np.ravel(y_data_train) # np.reshape(y_data_train, (y_data_train.shape[0], -1))
            for val in y_data_train_:
                writer.writerow(val)

    # sanity check on required directories
    if not os.path.exists('./output_files'):
        os.makedirs('./output_files')
    if not os.path.exists('./output_files/' + args.output_fname):
        os.makedirs('./output_files/' + args.output_fname)
    if not os.path.exists('./output_files/' + args.output_fname + '/csv'):
        os.makedirs('./output_files/' + args.output_fname + '/csv')

    return x_data_train, y_data_train, x_data_test, y_data_test

