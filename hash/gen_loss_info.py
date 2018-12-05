
import numpy as np


def gen_loss_info(
        train_info=None,
        cache_info=None):

    loss_type = train_info['hash_loss_type']

    if loss_type == 'Hinge':
        clac_loss_paris_fn = clac_loss_paris_Hinge

        gen_cache_fn = gen_cache_Hinge

    cache_info['clac_loss_paris_fn'] = clac_loss_paris_fn

    cache_info['gen_pair_weights_fn'] = gen_pair_weights

    cache_info = gen_cache_fn(train_info, cache_info)

    return cache_info


def gen_pair_weights(
        train_info=None,
        cache_info=None,
        init_hamm_dist_pairs0=None,
        loss_bit_num=None):

    clac_loss_paris_fn = cache_info['clac_loss_paris_fn']

    one_hamm_dist_pairs = init_hamm_dist_pairs0

    dist0_loss_items = clac_loss_paris_fn(
        train_info, cache_info, one_hamm_dist_pairs, loss_bit_num)

    one_hamm_dist_pairs = init_hamm_dist_pairs0 + 1

    dist1_loss_items = clac_loss_paris_fn(
        train_info, cache_info, one_hamm_dist_pairs, loss_bit_num)

    pair_weights = dist0_loss_items - dist1_loss_items

    return pair_weights, dist0_loss_items, dist1_loss_items


def clac_loss_paris_Hinge(
        train_info=None,
        cache_info=None,
        hamm_dist_pairs=None,
        bit_num=None):

    #    gt_hdist_norm_pairs = cache_info['gt_hdist_norm_pairs']

    gt_hdist_norm_pairs = cache_info['gt_hdist_norm_pairs']

    loss_pairs = (gt_hdist_norm_pairs.astype(float) -
                  hamm_dist_pairs.astype(float) / bit_num)

    relevant_sel = cache_info['relevant_sel'].reshape(-1,)
 #   relevant_sel = relevant_sel.astype('int')

    loss_pairs[relevant_sel] = - loss_pairs[relevant_sel]

    # hinge here:
    loss_pairs = np.maximum(loss_pairs, 0)

    loss_pairs = np.power(loss_pairs, 2)

    return loss_pairs


def gen_cache_Hinge(train_info=None, cache_info=None):

    # or change this setting

    hdist_pos_ratio = 0.0

    hdist_neg_ratio = 0.5

    assert(hdist_pos_ratio < hdist_neg_ratio)
    relevant_sel = cache_info['relevant_sel']
    # relevant_sel = relevant_sel.astype('int')

    gt_hdist_norm_pairs = np.ones(relevant_sel.shape)

    gt_hdist_norm_pairs[relevant_sel] = hdist_pos_ratio

    gt_hdist_norm_pairs[np.logical_not(relevant_sel)] = hdist_neg_ratio
    # gt_hdist_norm_pairs[gt_hdist_norm_pairs !=
    #                     hdist_pos_ratio] = hdist_neg_ratio

    cache_info['gt_hdist_norm_pairs'] = gt_hdist_norm_pairs

    return cache_info
