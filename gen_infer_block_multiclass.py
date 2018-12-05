import numpy as np


def gen_infer_block_multiclass(label_data=None):

    print('\n------------- gen_infer_block_multiclass... \n')

    assert (label_data.shape[1] == 1)
    e_num = label_data.shape[0]

    label_vs, new_y = np.unique(label_data, return_inverse=True)

    label_map = np.array([False]*e_num*len(label_vs)
                         ).reshape(e_num, len(label_vs))

    tmp_arr = np.arange(e_num * len(label_vs)).reshape(e_num, len(label_vs))
    for (new, i) in zip(new_y, range(e_num)):
        liner_index = tmp_arr[i, new]
        label_map.ravel()[liner_index] = 1

    relevant_groups = list()

    if not len(relevant_groups) > 0:
        relevant_groups = gen_relevant_groups_predifined(label_map)

    relevant_groups = list(relevant_groups)
    assert (len(relevant_groups) > 0)
    print('\n------------- gen_infer_block_multiclass finished \n')
    infer_info = dict()
    infer_info['infer_groups'] = relevant_groups

    return infer_info['infer_groups']


def gen_relevant_groups_predifined(group_map=None):

    g_num = group_map.shape[1]

    relevant_groups = np.array([None]*g_num).reshape(g_num,)

    valid_sel = np.array([False]*g_num).reshape(g_num,)

    for g_idx in xrange(g_num):
        one_group_idxes = list(np.nonzero(group_map[:, g_idx]))

        if one_group_idxes[0].size > 0:
            relevant_groups[g_idx] = one_group_idxes

            valid_sel[g_idx] = True

    relevant_groups = relevant_groups[valid_sel == True]

    return relevant_groups
