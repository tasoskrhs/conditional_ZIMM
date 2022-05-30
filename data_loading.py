
import scipy.io
import numpy as np

def load_data(data_fname,
              NoTest = 500 # number of Test samples
              ):

    #fname = 'input_data/' + data_fname
    #data = scipy.io.loadmat(fname + '.mat')
    data = scipy.io.loadmat(data_fname)
    data_tmp = np.array(data['data_swiss'])
    x_ = data_tmp[:, 0:2]  # GMM samples
    y_ = data_tmp[:, 2]  # samples's labels -> now they lie in [0, 1]

    # typecast to avoid issues later...
    x_ = x_.astype(dtype= np.float32)
    y_ = y_.astype(dtype= np.float32)

    # idx = np.random.randint(toy_data.shape[0], size=toy_data.shape[0])
    idx = np.random.randint(x_.shape[0], size=x_.shape[0])  # choose uniformly random integers in [0,x_.shape[0]) WITH resampling (possibly)
    i = int(x_.shape[0]) - NoTest

    x_data_train = x_[idx[:i], :]
    y_data_train = y_[idx[:i]]
    x_data_test = x_[idx[i:], :]
    y_data_test = y_[idx[i:]]

    return x_data_train, y_data_train, x_data_test, y_data_test


# Synthetic data example. Note that we provide the dir as data_fname and the .mat files have prespecified names
def load_synth_data(data_fname,
              NoTest = 3000, # number of Test samples
              missing_labels='none' # indicate which example case we choose for the training data set
              ):
    if missing_labels == 'none':
        data = scipy.io.loadmat(data_fname + 'expressions.mat')
        x_ = np.array(data['expressions'])
        # Condition (labels)
        data = scipy.io.loadmat(data_fname + 'time.mat')
        y_ = np.array(data['time_points'])  # labels: time-stamp (pseudotime) in [0, 1]

        # typecast
        x_ = x_.astype(dtype=np.float32)
        y_ = y_.astype(dtype=np.float32)
        y_ = np.ravel(y_)  # flatten array

        idx = np.random.randint(x_.shape[0],
                                size=x_.shape[0])  # choose uniformly random integers with possible resampling

        x_data_test = x_[idx[:NoTest], :]
        y_data_test = y_[idx[:NoTest]]

        x_data_train = x_[idx[NoTest:], :]  # shuffle
        y_data_train = y_[idx[NoTest:]]


    elif missing_labels == '0.4_0.6':
        data = scipy.io.loadmat(data_fname + 'expressions_in_0.4_0.6.mat')
        x_ = np.array(data['expr_interv'])
        # Condition (labels)
        data = scipy.io.loadmat(data_fname + 'times_in_0.4_0.6.mat')
        y_ = np.array(data['times_interv'])  # labels: time-stamp (pseudotime) in [0.4, 0.6]

        # typecast
        x_ = x_.astype(dtype=np.float32)
        y_ = y_.astype(dtype=np.float32)
        y_ = np.ravel(y_)  # flatten array

        idx = np.random.randint(x_.shape[0], size=x_.shape[0])
        x_data_test = x_[idx[:NoTest], :]
        y_data_test = y_[idx[:NoTest]]

        data = scipy.io.loadmat(data_fname + 'expressions_NOT_in_0.4_0.6.mat')
        x_remain = np.array(data['expr_remain'])
        # Condition (labels)
        data = scipy.io.loadmat(data_fname + 'times_NOT_in_0.4_0.6.mat')
        y_remain = np.array(data['times_remain'])  # labels: time-stamp (pseudotime) in [0, 0.4], [0.6, 1]

        # typecast
        x_remain = x_remain.astype(dtype=np.float32)
        y_remain = y_remain.astype(dtype=np.float32)
        y_remain = np.ravel(y_remain)  # flatten array

        idx = np.random.randint(x_remain.shape[0], size=x_.shape[0])
        x_data_train = x_remain[idx, :]  # shuffle
        y_data_train = y_remain[idx]

    elif missing_labels == 'state_2':
        data = scipy.io.loadmat(data_fname + 'expressions_Label_1.mat')
        x_lab1 = np.array(data['expressions'])
        y_lab1 = np.array(0.2*(np.ones(x_lab1.shape[0], dtype=np.float32))) # arbitrarily set label to '0.2'

        data = scipy.io.loadmat(data_fname + 'expressions_Label_3.mat')
        x_lab3 = np.array(data['expressions'])
        y_lab3 = np.array(0.8*(np.ones(x_lab3.shape[0], dtype=np.float32))) # arbitrarily set label to '0.8'

        # add 5% of state 2 as subpopulation
        data = scipy.io.loadmat(data_fname + 'expressions_Label_2.mat')
        x_lab2 = np.array(data['expressions'])
        x_lab2 = x_lab2[np.random.randint(x_lab2.shape[0], size=int(0.05* x_lab2.shape[0])), :]  # subsample at 5%
        y_lab2 = np.array(0.5 * (np.ones(x_lab2.shape[0], dtype=np.float32)))  # arbitrarily set label to '0.5'

        # typecast
        x_lab1 = x_lab1.astype(dtype=np.float32)
        x_lab2 = x_lab2.astype(dtype=np.float32)
        x_lab3 = x_lab3.astype(dtype=np.float32)

        # concatenate states 1, 2 & 3
        x_ = np.concatenate((x_lab1, x_lab2, x_lab3), axis= 0)
        y_ = np.concatenate((y_lab1, y_lab2, y_lab3), axis= 0)
        y_ = np.ravel(y_)  # np.reshape(y_, (-1, 1)) #flatten array

        idx = np.random.randint(x_.shape[0], size=x_.shape[0])
        x_data_train = x_[idx, :]  # shuffle
        y_data_train = y_[idx]

        # --- Test data ---
        data = scipy.io.loadmat(data_fname + 'expressions_Label_2.mat')
        x_lab2 = np.array(data['expressions'])
        y_lab2 = np.array(0.5 * (np.ones(x_lab2.shape[0], dtype=np.float32)))  # arbitrarily set label to '0.5'
        y_lab2 = np.ravel(y_lab2)  #  np.reshape(y_lab2, (-1, 1)) #flatten array
        idx = np.random.randint(x_lab2.shape[0], size=x_lab2.shape[0])

        # typecast
        x_lab2 = x_lab2.astype(dtype=np.float32)

        x_data_test = x_lab2[idx[:NoTest], :]
        y_data_test = y_lab2[idx[:NoTest]]

    else:
        raise ValueError("Invalid missing_labels case")


    return x_data_train, y_data_train, x_data_test, y_data_test


# Real mass cytometery data example. Note that we provide the dir as data_fname and the .mat files have prespecified names
def load_real_data(data_fname,
              NoTest = 1000 # number of Test samples
              ):
    data = scipy.io.loadmat(data_fname + 'markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16.mat')
    x_lab1 = np.array(data['data_H'])
    # typecast to avoid issues later...
    x_lab1 = x_lab1.astype(dtype=np.float32)
    x_H = x_lab1[np.random.randint(x_lab1.shape[0], size=26000), :]  # subsample at 26000 samples as reference
    y_H = np.array(0.2 * (np.ones(x_H.shape[0], dtype=np.float32)))  # arbitrarily set the label of the healthy class to '0.2'

    # ADD 1% of label CN as SUBPOPULATION...
    data = scipy.io.loadmat(data_fname + 'CN_all_markers.mat')
    x_lab2 = np.array(data['data_CN'])
    # typecast to avoid issues later...
    x_lab2 = x_lab2.astype(dtype=np.float32)
    x_CN_all = x_lab2[np.random.randint(x_lab2.shape[0], size=26000 + NoTest), :]  # subsample at 26000 samples as reference
    x_CN = x_CN_all[0:26000,:] # split train
    x_CN_test = x_CN_all[26000:,:] # split  test
    x_CN = x_CN[np.random.randint(x_CN.shape[0], size=int(0.01 * x_CN.shape[0])), :]  # subsample at 1%
    y_CN = np.array(0.5 * (np.ones(x_CN.shape[0], dtype=np.float32)))  # arbitrarily set the label of the diseased class to '0.5'

    # concatenate classes H & CN
    x_ = np.concatenate((x_H, x_CN), axis=0)
    y_ = np.concatenate((y_H, y_CN), axis=0)
    y_ = np.reshape(y_, (-1, 1))  # for the second dim to be one...

    idx = np.random.randint(x_.shape[0], size=x_.shape[
        0])  # choose uniformly random integers in [0,x_.shape[0]) with possible resampling

    x_data_train = x_[idx, :] # shuffle data
    y_data_train = y_[idx]

    x_data_test = x_CN_test
    y_data_test = np.array( 0.5 * (np.ones(x_CN_test.shape[0], dtype=np.float32)))  # label '0.5'
    y_data_test = np.reshape(y_data_test, (-1,1)) # for the second dim to be one...

    print('training data size=', x_data_train.shape)
    print('test data size=', x_data_test.shape)

    return x_data_train, y_data_train, x_data_test, y_data_test

