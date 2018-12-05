import cPickle as pickle
import os
import os.path as osp


def save_data(path, data):

    data_dir = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
