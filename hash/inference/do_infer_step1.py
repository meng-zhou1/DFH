
import numpy as np
import scipy.sparse as sp

from libs.pygco import pygco


def do_infer_step1(train_info=None, infer_info=None):

    # turn this on for debug
    infer_calc_obj = False

    init_bi_code = infer_info['init_bi_code']

    init_infer_result = dict()

    init_infer_result['infer_bi_code'] = init_bi_code

    if infer_calc_obj:
        obj_init = calc_infer_obj(init_bi_code, infer_info)

        init_infer_result['obj_value'] = obj_init

    infer_result = dict()

    if train_info['do_infer_block']:
        infer_result = do_infer_block(
            train_info, infer_info, init_infer_result)

    assert len(infer_result) > 0
    infer_result['obj_reduced'] = np.nan

    if infer_calc_obj:
        if 'obj_value' not in infer_result:
            infer_result['obj_value'] = calc_infer_obj(
                infer_result['infer_bi_code'], infer_info)

        infer_result['obj_reduced'] = obj_init - infer_result['obj_value']

    # assert(infer_result.reduced_obj>=-1e-6);

    bi_code = infer_result['infer_bi_code']

    assert (bi_code.shape[1] == 1)
    assert (bi_code.shape[0] == train_info['e_num'])
    assert bi_code.dtype == 'int'

    return infer_result


def calc_infer_obj(bi_code=None, infer_info=None):

    assert (len(bi_code) == infer_info['e_num'])
    relation_map = infer_info['relation_map']

    relation_weights = infer_info['relation_weights']

    relation_aff = calc_hamm_affinity(bi_code, relation_map)

    obj_value = sum(np.multiply(relation_aff, relation_weights))

    if infer_info['single_weights'].size > 0:
        obj_value = obj_value + \
            sum(np.multiply(infer_info['single_weights'], np.double(bi_code)))

    return obj_value


def gen_infer_result(infer_name=None, infer_bi_code=None):

    infer_result = dict()

    infer_result['infer_name'] = infer_name

    infer_result['infer_bi_code'] = infer_bi_code.astype(int)

    return infer_result


def idxsum(values=None, idxes=None, value_num=None):

    sum_v = np.bincount(idxes[:, 0], weights=values[:, 0])

    if len(sum_v) < value_num:
        sum_v[value_num] = 0

    return sum_v


def gen_relation_weight_block(
        sample_info=None,
        relation_weights=None,
        init_bi_code=None):

    sample_e_num = sample_info['sample_e_num']

    relation_weights = relation_weights[sample_info['sel_r_idxes']]

    non_sel_bi_code = init_bi_code[sample_info['non_sel1_e_idxes_other']]

    relation_weights = relation_weights.reshape(-1, 1)
    non_sel_weights = relation_weights[sample_info['non_sel1']
                                       ] * np.double(non_sel_bi_code)

    sw_extra_weights = idxsum(
        non_sel_weights,
        sample_info['non_sel1_e_idxes'],
        sample_e_num)

    if sample_info['non_sel2_e_idxes'].size > 0:
        non_sel_bi_code = init_bi_code[sample_info['non_sel2_e_idxes_other'].astype(
            int)]

        non_sel_weights = relation_weights[sample_info['non_sel2'].astype(
            int)] * np.double(non_sel_bi_code)

        sw_extra_weights2 = idxsum(
            non_sel_weights,
            sample_info['non_sel2_e_idxes'],
            sample_e_num)

        sw_extra_weights = sw_extra_weights + sw_extra_weights2

    single_weights = sw_extra_weights

    relation_weights = relation_weights[sample_info['multual_sel']]

    return relation_weights, single_weights


def update_infer_info_block(
        one_infer_info=None,
        infer_info=None,
        infer_bi_code=None):

    relation_weights = infer_info['relation_weights']

    relation_weights, single_weights = gen_relation_weight_block(
        one_infer_info['sample_info'], relation_weights, infer_bi_code)

    one_infer_info['relation_weights'] = relation_weights

    one_infer_info['single_weights'] = single_weights

    return one_infer_info


def do_infer_graphcut(
        train_info=None,
        infer_info=None,
        init_infer_result=None):

    e_num = infer_info['e_num']

    relation_map = infer_info['relation_map']

    relation_weights = infer_info['relation_weights']

    # submodular condition:
    if np.max(relation_weights) > np.spacing(1):
        print ('\n WARNING, submodularity is not satisfied...\n')
        relation_weights = np.minimum(relation_weights, 0)

    if not relation_map.dtype == 'uint64':
        relation_map = np.uint(np.double(relation_map))

    weight_mat_block = sp.csr_matrix(
        (-relation_weights[:, 0],
         (relation_map[:, 0], relation_map[:, 1])),
        shape=(e_num, e_num)).toarray()

    weight_mat_block = weight_mat_block + weight_mat_block.T
    single_weights = infer_info['single_weights'].reshape(1, e_num)

    assert (single_weights.size > 0)
    unary = np.concatenate((np.zeros((1, e_num)), single_weights), axis=0)

    label_pairwise_cost = np.ones((2, 2))
    label_pairwise_cost[0, 0] = 0
    label_pairwise_cost[1, 1] = 0

    conn_map = np.nonzero(weight_mat_block)

    init_label = np.zeros((1, e_num))
    init_infer_result['infer_bi_code'] = init_infer_result['infer_bi_code'].reshape(
        -1, 1)
    init_label[:, init_infer_result['infer_bi_code'][:, 0] > 0] = 1

    temp_map = np.ascontiguousarray(np.array(conn_map).T)
    unary = np.ascontiguousarray(unary.T)

    edge_weights = train_info['affinity_feature'][temp_map[:,
                                                           0], temp_map[:, 1]]

    if np.count_nonzero(conn_map) > 0:
        labels = pygco.cut_general_graph(
            temp_map,
            edge_weights,
            unary,
            label_pairwise_cost,
            algorithm='swap',
            init_labels=init_label[0])

    else:
        labels = np.argmin(unary, axis=0)
        labels = labels - 1

    infer_bi_code = np.ones((len(labels), 1))
    infer_bi_code[labels < 1] = -1
    infer_bi_code = infer_bi_code[0:infer_info['e_num']]

    infer_result = dict()
    infer_result['infer_bi_code'] = infer_bi_code

    return infer_result


def do_infer_block(
        train_info=None,
        infer_info=None,
        init_infer_result=None):

    infer_block_type = train_info['infer_block_type']

    if infer_block_type == 'graphcut':
        one_infer_fn = do_infer_graphcut

    assert one_infer_fn is not None
    infer_iter_num = train_info['infer_iter_num']

    infer_iter_counter = 0

    infer_info_groups = infer_info['infer_cache']['infer_info_groups']

    group_num = len(infer_info_groups)

    infer_bi_code = init_infer_result['infer_bi_code']

    while True:

        group_idxes = np.arange(group_num)

        for g_idx_idx in np.arange(group_num).reshape(-1):
            g_idx = group_idxes[g_idx_idx]

            one_infer_info = infer_info_groups[g_idx]

            one_infer_info = update_infer_info_block(
                one_infer_info, infer_info, infer_bi_code)

            one_init_infer_result = dict()

            one_init_infer_result['infer_bi_code'] = infer_bi_code[one_infer_info['sel_e_idxes']]

            one_infer_result = one_infer_fn(train_info,
                                            one_infer_info, one_init_infer_result)

            one_bi_code = one_infer_result['infer_bi_code']

            one_infer_info['sel_e_idxes'] = one_infer_info['sel_e_idxes'].reshape(
                1, one_infer_info['e_num'])

            infer_bi_code[one_infer_info['sel_e_idxes']] = one_bi_code

        infer_iter_counter = infer_iter_counter + 1

        if infer_iter_counter >= infer_iter_num:
            break

    infer_result = gen_infer_result(
        ('block_' + infer_block_type), infer_bi_code)

    infer_result['infer_iter_num'] = infer_iter_num

    return infer_result


def calc_hamm_affinity(bi_code=None, relation_map=None):

    relation_aff = np.ones((relation_map.shape[0], 1))

    not_identical_sel = bi_code[relation_map[:, 1]
                                ] != bi_code[relation_map[:, 2]]

    relation_aff[not_identical_sel] = - 1

    return relation_aff
