
import cPickle as pickle
import numpy as np

from evaluate.hash_evaluate import hash_evaluate
from evaluate.return_MAP import CalcMap
from hash.hash_encode import hash_encode
from hash.save_data import save_data
from libs.gcForest.settings.fg_scan import fg_scan
from run_DFH import run_DFH


if __name__ == '__main__':

    # choose dataset for training
    ds = dict()
    ds['name'] = 'mnist'
    ds['len'] = 28
    ds['wid'] = 28
    ds['channel'] = 1
    ds['tra_num'] = 5000
    ds['tst_num'] = 1000
    ds['phase'] = 'tra'

    ds_file = './datasets/{}.pkl'.format(ds['name'])

    print "load dataset: {} ".format(ds['name'])
    # load the demo dataset:

    with open(ds_file, 'rb') as f:
        ds_data = pickle.load(f)
    (X_train, y_train), (X_test, y_test) = ds_data

    train_data = dict()
    train_data['feat_data'] = X_train[:ds['tra_num']].reshape(
        ds['tra_num'], ds['channel'], ds['len'], ds['wid'])
    train_data['label_data'] = y_train[:ds['tra_num']]

    test_data = dict()
    test_data['feat_data'] = X_test[:ds['tst_num']:].reshape(
        ds['tst_num'], ds['channel'], ds['len'], ds['wid'])
    test_data['label_data'] = y_test[:ds['tst_num']]

    # retain the original data for generate affinity feature
    raw_data = dict()
    raw_data['feat_data'] = X_train[:ds['tra_num']].reshape(ds['tra_num'], -1)
    raw_data['label_data'] = y_train[:ds['tra_num']][:, None]

    # fine-grained scanning
    train_data['feat_data'] = fg_scan(train_data, ds)
    train_data['label_data'] = train_data['label_data'][:, None]

    # evaluation setting:
    # try larger bits: 32, 48, 64...
    bit_num = 8

    label_type = 'multiclass'

    train_label_info = dict()
    train_label_info['label_data'] = train_data['label_data']
    train_label_info['label_type'] = label_type

    test_label_info = dict()
    test_label_info['label_data'] = test_data['label_data']
    test_label_info['label_type'] = label_type

    code_data_info = dict()
    code_data_info['train_label_info'] = train_label_info
    code_data_info['test_label_info'] = test_label_info

    # eva_bit_step = round(min(8, bit_num / 4))
    eva_bit_step = 2
    eva_bits = np.arange(eva_bit_step, bit_num + 1, eva_bit_step)

    eva_param = dict()
    eva_param['eva_top_knn_pk'] = 100
    eva_param['eva_bits'] = eva_bits
    eva_param['code_data_info'] = code_data_info
    predict_results = []

    # run DFH:

    train_data_code = run_DFH(raw_data, train_data, eva_param, ds)
    print "training is finished"
    train_data_code = train_data_code > 0

    # save train data encoding result
    tra_data_code_path = "./models/{}/tra_data_code.pkl".format(ds['name'])
    save_data(tra_data_code_path, train_data_code)
    print "train encoding result save in {}".format(tra_data_code_path)

    ds['phase'] = 'tst'
    test_data_code = hash_encode(test_data, ds, bit_num)
    # save testdata encoding result
    test_data_code_path = "./models/{}/tst_data_code.pkl".format(ds['name'])
    save_data(test_data_code_path, test_data_code)
    print "test encoding result save in {}".format(test_data_code_path)

    print ('doing Deep Forest Hash evaluation...')

    eva_step = 2

    eva_bits = range(2, bit_num+1, eva_step)
    Map = []
    print 'Map:'
    for i in eva_bits:
        Map.append(CalcMap(test_data_code[:,:i], train_data_code[:,:i], test_data['label_data'], train_data['label_data']))
        print i, Map[-1]

    eva_bits = eva_param['eva_bits']

    code_data_info = eva_param['code_data_info']

    predict_result = dict()

    predict_result['method_name'] = 'DFH'

    predict_result['eva_bits'] = eva_bits

    pk100_eva_bits = [None]*eva_bits.size
    print 'Precision:'
    for b_idx in np.arange(eva_bits.size).reshape(-1):
        one_bit_num = int(eva_bits[b_idx])

        code_data_info = code_data_info

        code_data_info['train_data_code'] = train_data_code[:, 0:one_bit_num]

        code_data_info['tst_data_code'] = test_data_code[:, 0:one_bit_num]

        one_bit_result = hash_evaluate(eva_param, code_data_info)

        print (b_idx + 1) * 2, one_bit_result['pk100']
        pk100_eva_bits[b_idx] = one_bit_result['pk100']

    # save results
    precision_path = "./models/{}/result.pkl".format(ds['name'])

    with open(precision_path, 'wb') as f:
        pickle.dump(pk100_eva_bits, f, pickle.HIGHEST_PROTOCOL)

    map_path = "./models/{}/result.pkl".format(ds['name'])

    with open(map_path, 'wb') as f:
        pickle.dump(Map, f, pickle.HIGHEST_PROTOCOL)

    print "the final result save in {}".format(precision_path)

    print "\n------Finished|Finishded|Finishded------"

