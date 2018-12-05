
from gen_affinity_feature import gen_affinity_feature
from gen_affinity_labels import gen_affinity_labels
from gen_infer_block_multiclass import gen_infer_block_multiclass
from gen_relation_info import gen_relation_info
from hash.DFH_train import hash_train


def run_DFH(
        raw_data=None,
        train_data=None,
        eva_param=None,
        ds=None):
    """
        generate pairwise affinity ground truth for supervised hashing learning:
    a simple example is shown here for dataset with multi-class labels.
    users can replace this ground truth definition here according to your applications.
    this similarity labels can be cached.

    """

    print "\ngenerate affinity information..."
    trn_e_num = train_data['feat_data'].shape[0]

    affinity_labels = gen_affinity_labels(train_data)
    affinity_feature = gen_affinity_feature(raw_data)

    assert(affinity_labels.shape[0] == trn_e_num)
    assert(affinity_labels.shape[1] == trn_e_num)

    train_data['relation_info'] = gen_relation_info(affinity_labels)

    print "\nconstructing blocks for inference..."
    # generate inference blocks for block GraphCut.
    infer_block_info = gen_infer_block_multiclass(train_data['label_data'])

    print('\nconfiguration before training...')
    bit_num = eva_param['eva_bits'][-1]

    fasthash_train_info = dict()
    fasthash_train_info['affinity_feature'] = affinity_feature
    fasthash_train_info['bit_num'] = bit_num
    fasthash_train_info['binary_infer_method'] = 'block_graphcut'

    if fasthash_train_info['binary_infer_method'] == 'block_graphcut':
        fasthash_train_info['infer_info'] = infer_block_info

        fasthash_train_info['infer_iter_num'] = 2

    fasthash_train_info['hash_loss_type'] = 'Hinge'

    classifier_type = 'cascade_forest'

    hash_learner_param = dict()
    hash_learner_param['bit_num'] = bit_num
    hash_learner_param['classifier_type'] = classifier_type

    fasthash_train_info['hash_learner_param'] = hash_learner_param

    hash_code = hash_train(
        fasthash_train_info, train_data, ds)

    return hash_code
