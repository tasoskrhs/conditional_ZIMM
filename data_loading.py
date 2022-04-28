
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
    y_ = data_tmp[:, 2]  # samples's labels -> now they lie in (0, 1)

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