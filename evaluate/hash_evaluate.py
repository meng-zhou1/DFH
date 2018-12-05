
import numpy as np


def hash_evaluate(eva_param=None, code_data_info=None, *args, **kwargs):

    dist_info = gen_dist_info(
        code_data_info['tst_data_code'], code_data_info['train_data_code'], eva_param)
    train_label_info = code_data_info['train_label_info']
    test_label_info = code_data_info['test_label_info']
    cache_info = calc_agree_mat(dist_info, train_label_info, test_label_info)

    result_ir = np.array([])
    result_ir = do_eva_pk(eva_param, result_ir, cache_info)

    predict_result = result_ir

    return predict_result


def calc_agree_mat(dist_info=None, db_label_info=None, test_label_info=None, *args, **kwargs):
    dist_sort_idx_mat = dist_info['dist_sort_idx_mat']

    label_type = db_label_info['label_type']

    agree_mat = np.array([])

    if label_type == 'multiclass':
        Ytrain = db_label_info['label_data']
        Ytest = test_label_info['label_data']

        assert (np.unique(Ytrain).shape[0] < 2 ** 16)
        Ytrain = np.uint16(Ytrain)
        Ytest = np.uint16(Ytest)

        Labels = Ytrain[dist_sort_idx_mat]

        if dist_sort_idx_mat.shape[0] == 1:
            Labels = Labels.T

        temp_ytest = np.tile(Ytest.reshape(
            Ytest.shape[0], 1), (1, Labels.shape[1]))
        agree_mat = np.equal(temp_ytest, Labels.reshape(
            Labels.shape[0], Labels.shape[1]))

        cache_info = dict()
        cache_info['test_data_labels'] = Ytest
        cache_info['train_data_labels'] = Ytrain

    assert agree_mat.size > 0
    cache_info['agree_mat'] = agree_mat

    return cache_info


def do_eva_pk(eva_param=None, result_ir=None, cache_info=None):

    agree_mat = cache_info['agree_mat']

    train_num = agree_mat.shape[1]
    test_num = agree_mat.shape[0]

    test_k = eva_param['eva_top_knn_pk']
    test_k = min(test_k, train_num)

    eva_name = 'pk' + "{}".format(test_k)

    result_ir = dict()
    result_ir["{}".format(eva_name)] = np.mean(
        np.mean(agree_mat[:, 0:test_k], axis=1))
    return result_ir


def gen_dist_info(code_1=None, code_2=None, eva_param=None, *args, **kwargs):
    if 'use_weight_hamming' not in eva_param:
        eva_param['use_weight_hamming'] = False

    if eva_param['use_weight_hamming']:
        calc_hamming_dist_fn = calc_hamming_dist_weight

    else:
        calc_hamming_dist_fn = calc_hamming_dist

    trn_num = code_2.shape[0]
    test_num = code_1.shape[0]

    max_knn = 10000
    max_knn = min(max_knn, trn_num)

    large_data_thresh = 5000

    eva_capacity = np.ceil(large_data_thresh ** 2).astype(int)

    dist_mat = np.array([])

    if trn_num * test_num > eva_capacity:
        assert (trn_num < 2 ** 31)
        max_bit_num = code_1.shape[1] * 8

        assert (max_bit_num < 2 ** 15)
        dist_sort_idx_mat = np.zeros((test_num, max_knn), dtype=np.uint32)

        sort_dist_mat = np.zeros((test_num, max_knn), dtype=np.uint16)

        trn_step_size = min(trn_num, eva_capacity)

        tst_step_size = int(
            min(test_num, np.ceil(eva_capacity / trn_step_size)))

        tst_e_counter = 0

        while tst_e_counter < test_num:

            tst_start_idx = tst_e_counter + 1

            tst_end_idx = tst_e_counter + tst_step_size
            tst_end_idx = min(test_num, tst_end_idx)

            tst_sel_idxes = np.arange(tst_start_idx, tst_end_idx)

            one_test_num = tst_sel_idxes.size

            sel_test_one_dist = np.zeros(
                (one_test_num, trn_num), dtype=np.uint16)

            step_size = trn_step_size

            e_counter = 0

            sel_code_1_tmp = code_1[tst_sel_idxes, :]

            while e_counter < trn_num:

                start_idx = e_counter + 1

                end_idx = e_counter + step_size

                end_idx = min(trn_num, end_idx)

                sel_idxes = np.arange(start_idx, end_idx)

                sel_code_2_tmp = code_2[sel_idxes, :]

                if sel_code_1_tmp.shape[0] < sel_code_2_tmp.shape[0]:
                    one_one_dist = calc_hamming_dist_fn(
                        sel_code_1_tmp, sel_code_2_tmp, eva_param)

                else:
                    one_one_dist = calc_hamming_dist_fn(
                        sel_code_2_tmp, sel_code_1_tmp, eva_param)
                    one_one_dist = one_one_dist.T

                sel_test_one_dist[:, sel_idxes] = np.uint16(one_one_dist)

                e_counter = end_idx

            tst_e_counter = tst_end_idx

            one_sort_dist_mat = np.sort(sel_test_one_dist, axis=1)
            one_I = np.argsort(sel_test_one_dist, axis=1)

            one_I = one_I[:, 0:max_knn]

            one_sort_dist_mat = one_sort_dist_mat[:, 0:max_knn]

            dist_sort_idx_mat[tst_sel_idxes, :] = np.uint32(one_I)

            sort_dist_mat[tst_sel_idxes, :] = np.uint16(one_sort_dist_mat)

    else:
        dist_mat = calc_hamming_dist_fn(code_1, code_2, eva_param)

        dist_mat = np.uint16(dist_mat)

        sort_dist_mat = np.sort(dist_mat, axis=1)
        dist_sort_idx_mat = np.argsort(dist_mat, axis=1)

        if dist_sort_idx_mat.shape[1] > max_knn:
            dist_sort_idx_mat = dist_sort_idx_mat[:, 1:max_knn]

            sort_dist_mat = sort_dist_mat[:, 0:max_knn]

    dist_info = dict()
    dist_info['dist_mat'] = dist_mat
    dist_info['dist_sort_idx_mat'] = dist_sort_idx_mat
    dist_info['sort_dist_mat'] = sort_dist_mat

    return dist_info


def calc_hamming_dist(code_1=None, code_2=None, eva_param=None):
    assert code_1.dtype == 'bool'
    assert code_2.dtype == 'bool'
    e_num1 = code_1.shape[0]

    e_num2 = code_2.shape[0]

    assert (code_1.shape[1] < 2 ** 15)
    dist = np.zeros((e_num1, e_num2), dtype=np.uint16)

    for e_ind in np.arange(e_num1).reshape(-1):
        one_pair_feat = np.logical_xor(code_1[e_ind, :], code_2)

        assert one_pair_feat.dtype == 'bool'
        one_dist = np.sum(one_pair_feat, axis=1)

        dist[e_ind, :] = np.uint16(one_dist)

    return dist


def calc_hamming_dist_weight(code_1=None, code_2=None, eva_param=None):
    assert code_1.dtype == 'bool'
    assert code_2.dtype == 'bool'
    e_num1 = code_1.shape[0]

    e_num2 = code_2.shape[0]

    w = eva_param['hamming_weight']

    dist = np.zeros(e_num1, e_num2, 'single')

    for e_ind in np.arange(1, e_num1).reshape(-1):
        one_pair_feat = np.logical_xor(code_1[e_ind, :], code_2)

        if len(w) > one_pair_feat.shape[1]:
            w = w[1:one_pair_feat.shape[1]]

        one_dist = one_pair_feat * w

        dist[e_ind, :] = np.single(one_dist)

    return dist
