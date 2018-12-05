
import numpy as np
import scipy.sparse as sp

from hash.config_train_info import config_train_info
from hash.gen_loss_info import gen_loss_info
from hash.gen_relation_map import gen_relation_map
from hash.inference.gen_infer_info import gen_infer_info
from hash.train_step1 import train_step1
from hash.train_step2 import train_step2


def hash_train(train_info=None, train_data=None, ds=None):
    print('\n\n------------------------------Deep-Forest-Hash Training---------------------------\n\n')
    if 'train_id' not in train_info:
        train_info['train_id'] = 'Deep-Forest-Hash'

    train_info = config_train_info(train_info)

    train_info['e_num'] = train_data['feat_data'].shape[0]

    hash_code = do_train(train_info, train_data, ds)

    print('\n\n------------------------------Deep-Forest-Hash Training Finished---------------------------\n\n')

    return hash_code


def do_train(train_info=None, train_data=None, ds=None):

    relation_info = train_data['relation_info']

    work_info_step1 = gen_work_info_step1(relation_info)

    work_info_step1 = gen_loss_info(train_info, work_info_step1)

    work_info_step1['init_infer_info'] = gen_infer_info(
        train_info, work_info_step1)

    work_info_step1['data_weights'] = np.array([])

    work_info_step2 = dict()

    work_info_step1, work_info_step2, hash_code = do_train_stage(
        train_info, train_data, work_info_step1, work_info_step2, ds)

    return hash_code


def do_train_stage(
        train_info=None,
        train_data=None,
        work_info_step1=None,
        work_info_step2=None,
        ds=None):
    train_info['use_data_weight'] = True

    if not (train_info['train_stagewise']):
        bi_code_bits = np.ones(
            (train_info['e_num'],
             train_info['bit_num']),
            dtype=np.int8)

        train_info['use_data_weight'] = False

    run_converge = False

    update_bit = 0

    while not run_converge:

        update_bit = update_bit + 1

        work_info_step1['update_bi_code'] = np.array([])

        work_info_step2['update_bi_code_step1'] = np.array([])

        work_info_step1['update_bit'] = update_bit

        if not (train_info['train_stagewise']):
            work_info_step1['update_bit_loss'] = train_info['bit_num']

        else:
            work_info_step1['update_bit_loss'] = update_bit

        if update_bit > 1:
            pre_bi_code = work_info_step2['update_bi_code_step2']

            assert (len(pre_bi_code) > 0)
            one_hamm_dist_pairs = np.int8(calc_hamm_dist_r_map(
                pre_bi_code, work_info_step1['relation_map']))

            init_hamm_dist_pairs0 = np.int8(
                work_info_step1['init_hamm_dist_pairs0'])

            init_hamm_dist_pairs0 = sp.csr_matrix(
                init_hamm_dist_pairs0, dtype=np.int8)
            one_hamm_dist_pairs = sp.csr_matrix(
                one_hamm_dist_pairs.reshape(-1, 1), dtype=np.int8)
            init_hamm_dist_pairs0 = (
                init_hamm_dist_pairs0 + one_hamm_dist_pairs).toarray()

            work_info_step1['init_hamm_dist_pairs0'] = init_hamm_dist_pairs0

        work_info_step1 = train_step1(train_info, work_info_step1)

        disp_loop_info_step1(train_info, work_info_step1)

        work_info_step2['update_bi_code_step2'] = np.array([])

        assert (work_info_step1['update_bi_code'].shape > 0)
        if not (train_info['train_stagewise']):
            bi_code_bits[:, update_bit] = work_info_step1['update_bi_code']

            work_info_step2['update_bi_code_step2'] = work_info_step1['update_bi_code']

        else:
            work_info_step2['update_bit'] = update_bit

            work_info_step2['data_weights'] = work_info_step1['data_weights']

            work_info_step2['update_bi_code_step1'] = work_info_step1['update_bi_code']

            work_info_step2 = train_step2(
                train_info, train_data, work_info_step2, ds)

            # print update_bit

            if update_bit == 1:
                hash_code = work_info_step2['update_bi_code_step2']
            else:
                hash_code = np.column_stack(
                    (hash_code, work_info_step2['update_bi_code_step2']))

            disp_loop_info_step2(train_info, work_info_step2)
        if update_bit >= train_info['bit_num']:
            run_converge = True

    if not (train_info['train_stagewise']):
        for b_idx in np.arange(train_info['bit_num']).reshape(-1):
            work_info_step2['update_bit'] = b_idx

            work_info_step2['data_weights'] = np.array([])

            work_info_step2['update_bi_code_step1'] = bi_code_bits[:, b_idx]

            work_info_step2 = train_step2(
                train_info, train_data, work_info_step2)

            disp_loop_info_step2(train_info, work_info_step2)

    return work_info_step1, work_info_step2, hash_code


def gen_train_result(
        train_info=None,
        work_info_step1=None,
        work_info_step2=None):
    train_result = np.array([])
    raise NotImplementedError()


def clean_hash_learners(hash_learner_infos=None):

    bit_num = hash_learner_infos.shape[0]

    hash_learners = np.array((bit_num, 1), dtype=object)

    return hash_learners


def gen_work_info_step1(relation_info=None):

    work_info_step1 = dict()

    gen_param = dict()

    gen_param['relation_info'] = relation_info

    gen_result = gen_relation_map(gen_param)

    work_info_step1['relation_map'] = gen_result['relation_map']

    work_info_step1['relevant_sel'] = gen_result['relevant_sel']

    work_info_step1['init_hamm_dist_pairs0'] = np.zeros(
        (len(work_info_step1['relevant_sel']), 1))

    return work_info_step1


def calc_hamm_dist_r_map(bi_code=None, relation_map=None, *args, **kwargs):

    e_bi_code = bi_code[relation_map[:, 0]]

    right_bi_code = bi_code[relation_map[:, 1]]

    cmpre = e_bi_code != right_bi_code
    hamm_dist_pairs = np.sum(cmpre.reshape(-1, 1), axis=1)

    return hamm_dist_pairs


def disp_loop_info_step1(
        train_info=None,
        work_info_step1=None):

    print "\n---Step-1, train_id:{}, loss:{}, infer:{}, sw:{}, obj:{:.4f}, update_bit:{}/{}\n".format(
        train_info['train_id'],
        train_info['hash_loss_type'],
        train_info['binary_infer_method'],
        train_info['train_stagewise'],
        work_info_step1['obj_value'],
        work_info_step1['update_bit'],
        train_info['bit_num'])

    return


def disp_loop_info_step2(
        train_info=None,
        work_info_step2=None):
    update_bit = work_info_step2['update_bit']

    print "\n---Step-2, classifier:{}, update_bit:{}/{}, acc:{:.4f} \n".format(
        train_info['hash_learner_param']['classifier_type'],
        work_info_step2['update_bit'],
        train_info['bit_num'],
        work_info_step2['accuracy'])
    return
