import numpy as np


def gen_relation_info(affinity_labels=None):

    e_num = affinity_labels.shape[0]

    rel_mat = affinity_labels > 0
    irrel_mat = affinity_labels < 0

    rel_mat = np.logical_or(rel_mat, rel_mat.T)
    irrel_mat = np.logical_or(irrel_mat, irrel_mat.T)

    relation_info = dict()
    relation_info['rel_mat'] = rel_mat
    relation_info['irrel_mat'] = irrel_mat
    relation_info['e_num'] = e_num
    relation_info['get_rel_idxes_fn'] = get_rel_idxes
    relation_info['get_irrel_idxes_fn'] = get_irrel_idxes

    return relation_info


def get_rel_idxes(relation_info=None, e_idx=None):

    one_rel = relation_info['rel_mat'][:, e_idx]

    rel_idxes = np.nonzero(one_rel)

    return rel_idxes


def get_irrel_idxes(relation_info=None, e_idx=None):

    irrel_mat = relation_info['irrel_mat']

    if not irrel_mat.size > 0:
        one_rel = relation_info['rel_mat'][:, e_idx]
        one_rel[e_idx] = 0

        irrel_idxes = np.nonzero(np.logical_not(one_rel))

    else:
        irrel_idxes = np.nonzero(irrel_mat[:, e_idx])

    return irrel_idxes
