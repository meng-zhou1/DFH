import numpy as np


def gen_infer_info(train_info=None, work_info_step1=None, *args, **kwargs):

    infer_info = dict()

    infer_info['relation_map'] = work_info_step1['relation_map']

    infer_info['e_num'] = train_info['e_num']

    infer_info['infer_cache'] = gen_infer_cache(train_info, infer_info)

    if train_info['do_infer_spectral']:
        infer_info['relation_map'] = np.double(infer_info['relation_map'])

    return infer_info


def gen_infer_cache(train_info=None, infer_info=None, *args, **kwargs):

    infer_cache = dict()

    if train_info['do_infer_block']:
        infer_cache = gen_infer_cache_block(
            train_info, infer_info, infer_cache)

    return infer_cache


def gen_infer_cache_block(
        train_info=None,
        infer_info=None,
        infer_cache=None):
    # infer_groups = train_info['infer_info']['infer_groups']
    infer_groups = train_info['infer_info']

    group_num = len(infer_groups)

    relation_map = infer_info['relation_map']

    r1_org_global = relation_map[:, 0]

    r2_org_global = relation_map[:, 1]

    trans_map = np.zeros((infer_info['e_num'], 1), dtype=np.uint32)

    assert (len(r1_org_global) < 2.0 ** 31)
    shared_task_data = dict()

    shared_task_data['r1_org_global'] = r1_org_global

    shared_task_data['r2_org_global'] = r2_org_global

    shared_task_data['trans_map'] = trans_map

    shared_task_data['infer_groups'] = infer_groups

    if 'use_mmat' not in train_info:
        train_info['use_mmat'] = False

    if train_info['use_mmat']:
        task_inputs = list(np.arange(group_num))
#        task_inputs = num2cell((np.arange(1, group_num)).T, 2)

        task_num = len(task_inputs)

        mmat = train_info['mmat']


    else:
        infer_info_groups = [None]*group_num

        for g_idx in np.arange(group_num).reshape(-1):
            one_infer_info = do_one_task_gen_infer_cache(
                g_idx, shared_task_data)

            infer_info_groups[g_idx] = one_infer_info

    infer_cache['infer_info_groups'] = infer_info_groups

    return infer_cache


def do_one_task_gen_infer_cache_func(
        task_input=None,
        shared_task_data=None,
        runner_info=None,
        task_index=None):

    task_result = do_one_task_gen_infer_cache(task_input, shared_task_data)

    return task_result, shared_task_data


def do_one_task_gen_infer_cache(
        task_input=None,
        shared_task_data=None):

    g_idx = task_input

    r1_org_global = shared_task_data['r1_org_global']

    r2_org_global = shared_task_data['r2_org_global']

    trans_map = shared_task_data['trans_map']

    infer_groups = shared_task_data['infer_groups']

    sel_e_idxes = infer_groups[g_idx]

    sel_e_idxes = np.uint32(sel_e_idxes)
#    sel_e_idxes = np.where(sel_e_idxes, sel_e_idxes, 0)

    r1_sel = np.isin(r1_org_global, sel_e_idxes)

    r2_sel = np.isin(r2_org_global, sel_e_idxes)

    r_sel = np.logical_or(r1_sel, r2_sel)

    r1 = r1_org_global[r_sel]

    r2 = r2_org_global[r_sel]

    r1_sel = r1_sel[r_sel]

    r2_sel = r2_sel[r_sel]

    sel_r_idxes = np.uint32(np.nonzero(r_sel))

    exchange_sel = np.logical_not(r1_sel)

    tmp_r1 = r1[exchange_sel]

    r1[exchange_sel] = r2[exchange_sel]

    r2[exchange_sel] = tmp_r1

    r2_sel[exchange_sel] = False

    multual_sel = r2_sel

    trans_map[sel_e_idxes[0, :], 0] = np.arange(sel_e_idxes.shape[1])

    r2_org = r2

    r1 = trans_map[r1]

    sel_r1 = r1[multual_sel]

    sel_r2 = r2[multual_sel]

    sel_r2 = trans_map[sel_r2]

    sample_info = dict()

    sample_info['multual_sel'] = multual_sel

    non_sel1 = np.logical_not(multual_sel)

    sample_info['non_sel1'] = non_sel1

    sample_info['non_sel2'] = np.array([])

    sample_info['non_sel1_e_idxes'] = r1[non_sel1]

    sample_info['non_sel1_e_idxes_other'] = r2_org[non_sel1]

    sample_info['non_sel2_e_idxes'] = np.array([])

    sample_info['non_sel2_e_idxes_other'] = np.array([])

    sample_info['sample_e_num'] = sel_e_idxes.shape[1]

    sample_info['sel_r_idxes'] = sel_r_idxes

    one_relation_map = np.concatenate((sel_r1, sel_r2), axis=1)

    assert one_relation_map.dtype == 'uint32'
    one_infer_info = dict()

    one_infer_info['sample_info'] = sample_info

    one_infer_info['sel_e_idxes'] = sel_e_idxes

    one_infer_info['e_num'] = sel_e_idxes.shape[1]

    one_infer_info['relation_map'] = one_relation_map

    return one_infer_info
