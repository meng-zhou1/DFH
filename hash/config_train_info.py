
def config_train_info(train_info=None):

    # check parameter settings...

    if 'hash_loss_type' not in train_info:
        train_info['hash_loss_type'] = 'KSH'
        # train_info.hash_loss_type='Hinge';

    if 'train_stagewise' not in train_info:
        train_info['train_stagewise'] = True

    if 'binary_infer_method' not in train_info:
        train_info['binary_infer_method'] = 'block_graphcut'

    train_info['do_infer_spectral'] = False

    train_info['do_infer_block'] = False

    if train_info['binary_infer_method'] == 'block_graphcut':
        train_info['do_infer_block'] = True

        train_info['infer_block_type'] = 'graphcut'

    if train_info['binary_infer_method'] == 'spectral':
        train_info['do_infer_spectral'] = True

    return train_info
