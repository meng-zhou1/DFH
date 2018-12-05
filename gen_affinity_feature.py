import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize


def gen_affinity_feature(train_data=None):

    print "gen_affinity_feature..."
    e_num = train_data['feat_data'].shape[0]
    X = train_data['feat_data']
    y = train_data['label_data']

    X = normalize(X, norm='l2')

    dist = DistanceMetric.get_metric('euclidean')
    eud_dist = dist.pairwise(X)

    alpha = 1
    belta = 1
    gamma = 1

    dist_mat = np.zeros((eud_dist.shape))

    for e_idx in xrange(e_num):
        relevant_sel = y[e_idx] == y
        irrelevant_sel = np.logical_not(relevant_sel)
        relevant_sel[e_idx] = False

        relevant_idxes = (np.nonzero(relevant_sel))
        irrelevant_idxes = (np.nonzero(irrelevant_sel))

        dist_mat[e_idx, relevant_idxes] = np.sqrt(
            1 - np.exp(-eud_dist[e_idx, relevant_idxes] ** 2) / belta)
        dist_mat[e_idx, irrelevant_idxes] = np.sqrt(
            np.exp(eud_dist[e_idx, irrelevant_idxes] ** 2 / belta)) - alpha

    affinity_feature = np.exp((-dist_mat ** 2) / gamma)

    return affinity_feature
