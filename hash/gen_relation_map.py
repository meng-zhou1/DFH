import numpy as np


def gen_relation_map(gen_param=None):

    gen_result = gen_relation_map_duplet(gen_param)

    if gen_result['relation_map'].dtype == 'uint32':
        return gen_result


def gen_relation_map_duplet_simple(gen_param=None):

    relation_info = gen_param['relation_info']

    rel_mat = relation_info['rel_mat']

    sel_rel_mat = rel_mat

    r2_org = np.nonzero(sel_rel_mat)[0]
    r1 = np.nonzero(sel_rel_mat)[1]

    r2_org = np.uint32(r2_org)
    r1 = np.uint32(r1)

    valid_e_num = rel_mat.shape[0]

    rel_r_num = len(r1)

    irrel_mat = relation_info['irrel_mat']

    if irrel_mat.size > 0:
        sel_rel_mat = irrel_mat

        irsel_r2_org = np.nonzero(sel_rel_mat)[0]
        irsel_r1 = np.nonzero(sel_rel_mat)[1]

    else:
        irsel_r1 = [None]*valid_e_num

        irsel_r2_org = [None]*valid_e_num

        for e_idx_idx in np.arange(valid_e_num).reshape(-1):
            e_idx = e_idx_idx

            n_top_knn_inds = relation_info['get_irrel_idxes_fn'](e_idx)

            one_irsel_r2_org = np.uint32(n_top_knn_inds)

            one_irsel_r1 = np.tile(np.uint32(e_idx), (one_r_num, 1))

            irsel_r2_org[e_idx_idx] = one_irsel_r2_org

            irsel_r1[e_idx_idx] = one_irsel_r1

        irsel_r1 = np.array(irsel_r1)

        irsel_r2_org = np.array(irsel_r2_org)

    irrel_r_num = irsel_r1.size

    assert(rel_r_num > 0 or irrel_r_num)
    r1 = np.concatenate((r1, np.uint32(irsel_r1)), axis=0)

    r1_org = r1

    r2_org = np.concatenate((r2_org, np.uint32(irsel_r2_org)), axis=0)

    r_num = rel_r_num + irrel_r_num

    r1_org = r1_org[:, None]
    r2_org = r2_org[:, None]
    relation_map = np.hstack([r1_org, r2_org])

    relevant_sel = np.array([False]*r_num*1).reshape(r_num, 1)

    relevant_sel[0:rel_r_num] = 1

    assert(r_num < 2 ** 32)
    gen_result = dict()

    gen_result['relation_map'] = relation_map

    gen_result['relevant_sel'] = relevant_sel

    gen_result['r1_new'] = r1

    return gen_result


def gen_relation_map_duplet(gen_param=None):

    gen_result_simple = gen_relation_map_duplet_simple(gen_param)

    relation_map = gen_result_simple['relation_map']

    relevant_sel = gen_result_simple['relevant_sel']

    r1 = gen_result_simple['r1_new']

    r1_org = relation_map[:, 0]

    r2_org = relation_map[:, 1]

    r_num = r1.size

    valid_r_sel = r1_org > r2_org

    expect_r_num = r_num / 2

    relation_map = relation_map[valid_r_sel, :]

    relevant_sel = relevant_sel[valid_r_sel, :]

    assert(relation_map.shape[0] == expect_r_num)
    gen_result = dict()

    gen_result['relation_map'] = relation_map

    gen_result['relevant_sel'] = relevant_sel

    return gen_result
