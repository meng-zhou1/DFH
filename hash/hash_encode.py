
import os
import os.path as osp
import cPickle as pickle

import numpy as np


def hash_encode(data=None, ds=None, bit_num=None):

    print('\n\n------------------------------fasthash_encode---------------------------\n\n')

    print "fine-grained scanning for {}_data".format(ds['phase'])

    fg_model_path = "./models/{}/fg/fg_model.pkl".format(ds['name'])

    with open(fg_model_path, "rb") as f:
        gc = pickle.load(f)
    fg_result = gc.transform(data['feat_data'])

    fg_result_path = "./models/{}/fg/fg_{}_result.pkl".format(
        ds['name'], ds['phase'])
    with open(fg_result_path, "wb") as f:
        pickle.dump(fg_result, f, pickle.HIGHEST_PROTOCOL)

    # load fg_result if it has been trained
    # fg_result_path = "./models/{}/fg/fg_{}_result.pkl".format(ds['name'], ds['phase'])
    # with open(fg_result_path, 'rb') as f:
    #     fg_result = pickle.load(f)

    print "cascade classification for {}_data".format(ds['phase'])

    feat_data_code = [None]*bit_num
    for i in xrange(bit_num):
        ca_model_path = "./models/{}/fg-ca".format(ds['name'])
        search = SearchFile()
        file_name = 'ca_'+"{}.pkl".format(i+1)
        hash_func_path = search.findfile(file_name, ca_model_path)

        with open(hash_func_path, "rb") as f:
            gc = pickle.load(f)
        y_pred = gc.predict(fg_result)
        if i == 0:
            feat_data_code = y_pred
        else:
            feat_data_code = np.column_stack((feat_data_code, y_pred))

    # # cascade only
    # print "cascade classification for {}_data".format(ds['phase'])
    #
    # feat_data_code = [None]*bit_num
    # for i in xrange(bit_num):
    #     ca_model_path = "./models/{}/ca".format(ds['name'])
    #     search = SearchFile()
    #     file_name = 'ca_'+"{}.pkl".format(i+1)
    #     hash_func_path = search.findfile(file_name, ca_model_path)
    #
    #     with open(hash_func_path, "rb") as f:
    #         gc = pickle.load(f)
    #     y_pred = gc.predict(data['feat_data'])
    #     if i == 0:
    #         feat_data_code = y_pred
    #     else:
    #         feat_data_code = np.column_stack((feat_data_code, y_pred))

    feat_data_code = feat_data_code > 0

    return feat_data_code


class SearchFile(object):

    def __init__(self, path='.'):
        self._path = path
        self.abspath = os.path.abspath(self._path)

    def findfile(self, keyword, root):
        filelist = []
        for root, dirs, files in os.walk(root):
            for name in files:
                fitfile = filelist.append(os.path.join(root, name))
        for i in filelist:
            if os.path.isfile(i):
                if keyword in os.path.split(i)[1]:
                    return i
