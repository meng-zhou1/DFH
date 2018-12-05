
import numpy as np

from libs.gcForest.settings.cascade_forest import cascade_forest


def train_step2(
        train_info=None,
        train_data=None,
        work_info_step2=None,
        ds=None):

    update_bit = work_info_step2['update_bit']

#     hash_learner_cache_info = work_info_step2['hash_learner_cache_info']

    bi_train_data = dict()

    bi_train_data['feat_data'] = train_data['feat_data']
    bi_train_data['label_data'] = work_info_step2['update_bi_code_step1']
    bi_train_data['hash_learner_idx'] = update_bit
    bi_train_data['data_weight'] = work_info_step2['data_weights']

# reshape the data
    gcf_train_data = dict()
    gcf_train_data['feat_data'] = bi_train_data['feat_data']
    gcf_train_data['label_data'] = bi_train_data['label_data'].reshape(-1,)

    fg_label = train_data['label_data'].reshape(
        train_data['label_data'].shape[0],)

# train with gcforest
    hlearner_bi_code = cascade_forest(gcf_train_data, fg_label, update_bit, ds)

    work_info_step2['update_bi_code_step2'] = hlearner_bi_code

    acc = calc_accuracy(
        hlearner_bi_code, bi_train_data['label_data'], bi_train_data['data_weight'])
    work_info_step2['accuracy'] = acc
    return work_info_step2


def calc_accuracy(
        predict_labels=None,
        gt_labels=None,
        data_weight=None):

    gt_labels[gt_labels == -1] = 0
    gt_labels = gt_labels.reshape(gt_labels.shape[0],)
    correct_sel = gt_labels == predict_labels

    acc = float(np.count_nonzero(correct_sel)) / float(correct_sel.shape[0])

    return acc
