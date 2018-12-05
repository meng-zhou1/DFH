"""
generate pairwise affinity ground truth for supervised hashing learning:
a simple example is shown here for dataset with multi-class labels.
users can replace this ground truth definition here according to your
applications.

"""

import random

import numpy as np


def gen_affinity_labels(train_data=None):

    print ('gen_affinity_labels...')

    e_num = train_data['feat_data'].shape[0]

    label_data = train_data['label_data']

    assert(label_data.shape[0] == e_num)
    assert(label_data.shape[1] == 1)
    affinity_labels = np.zeros((e_num, e_num), dtype=np.int8)
    affinity_feature = np.zeros((e_num, e_num))

    # max_similar_num = 100
    # max_dissimilar_num = 100

    max_similar_num = float('inf')
    max_dissimilar_num = float('inf')

    for e_idx in np.arange(e_num):
        relevant_sel = label_data[e_idx] == label_data

        irrelevant_sel = np.logical_not(relevant_sel)

        relevant_sel[e_idx] = False

        relevant_idxes = list(np.nonzero(relevant_sel))

        if relevant_idxes[0].size > max_similar_num:
            relevant_idxes[0] = random.sample(
                relevant_idxes[0], max_similar_num)

        irrelevant_idxes = list(np.nonzero(irrelevant_sel))

        if irrelevant_idxes[0].size > max_dissimilar_num:
            irrelevant_idxes[0] = random.sample(
                irrelevant_idxes[0], max_dissimilar_num)

        affinity_labels[e_idx, relevant_idxes[0]] = 1
        affinity_labels[e_idx, irrelevant_idxes[0]] = - 1

        affinity_labels[relevant_idxes[0], e_idx] = 1
        affinity_labels[irrelevant_idxes[0], e_idx] = - 1

        affinity_labels[e_idx, e_idx] = 0

    return affinity_labels
