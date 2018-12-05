

import numpy as np

from hash.inference.do_infer_step1 import do_infer_step1


def train_step1(train_info=None, work_info_step1=None):

    init_hamm_dist_pairs0 = work_info_step1['init_hamm_dist_pairs0']

    infer_info = work_info_step1['init_infer_info']

    gen_pair_weights_fn = work_info_step1['gen_pair_weights_fn']

    update_bit_loss = work_info_step1['update_bit_loss']

    relation_weights, hamm_items0, ___ = gen_pair_weights_fn(
        train_info, work_info_step1, init_hamm_dist_pairs0, update_bit_loss)

    infer_info['relation_weights'] = relation_weights

    infer_info['init_bi_code'] = np.ones((train_info['e_num'], 1))

    infer_result = do_infer_step1(train_info, infer_info)

    work_info_step1['obj_value'] = np.mean(hamm_items0)

    work_info_step1['obj_reduced'] = infer_result['obj_reduced']

    work_info_step1['update_bi_code'] = infer_result['infer_bi_code']

    if train_info['use_data_weight']:
        work_info_step1 = updata_data_weights(work_info_step1, infer_info)

    return work_info_step1


def updata_data_weights(
        work_info_step1=None,
        infer_info=None):

    r_w_nosign = abs(infer_info['relation_weights'])

    relation_map = work_info_step1['relation_map']

    r_w_nosign = r_w_nosign.reshape(-1, 1)
    work_info_step1['data_weights'] = np.bincount(
        np.ravel(relation_map, order='F'),
        weights=np.concatenate((r_w_nosign, r_w_nosign), axis=0)[:, 0])

    return work_info_step1
