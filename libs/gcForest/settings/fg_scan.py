
from hash.save_data import save_data
from libs.gcForest.lib.gcforest.gcforest import GCForest
from libs.gcForest.lib.gcforest.utils.config_utils import load_json


def fg_scan(train_data=None, ds=None):

    X_train = train_data['feat_data']
    y_train = train_data['label_data']

    fg_model = "./libs/gcForest/settings/{}/{}-fg.json".format(
        ds['name'], ds['name'])

    print '\nfine-grained scanning phase '
    config = load_json(fg_model)
    gc = GCForest(config)

    X_train_enc = gc.fit_transform(X_train, y_train)
    print '\nfine-grained scanning finished'

    # dump the fine-grained scanning model
    fg_model_path = "./models/{}/fg/fg_model.pkl".format(ds['name'])
    save_data(fg_model_path, gc)

    # dump the scanning result of fine-grained model
    fg_result_path = "./models/{}/fg/fg_{}_result.pkl".format(
        ds['name'], ds['phase'])
    save_data(fg_result_path, X_train_enc)
    print '\nfine-grained results saved'

    print 'feature fusion'
    final_feature = feature_fusion(X_train_enc)

    return final_feature


def feature_fusion(feature=None):

    W1 = 0.5
    W2 = 0.5

    ets1_feature = feature[0]
    ets2_feature = feature[2]
    ets3_feature = feature[4]

    rf1_feature = feature[1]
    rf2_feature = feature[3]
    rf3_feature = feature[5]

    win1_feature = W1 * ets1_feature + W2 * rf1_feature
    win2_feature = W1 * ets2_feature + W2 * rf2_feature
    win3_feature = W1 * ets3_feature + W2 * rf3_feature

    num, __, i2, j2 = win2_feature.shape
    __, __, i3, j3 = win3_feature.shape

    temp = win2_feature
    temp = temp + win1_feature[:, :, :i2, :j2]
    temp = temp + win1_feature[:, :, :i2, -j2:]
    temp = temp + win1_feature[:, :, -i2:, :j2]
    temp = temp + win1_feature[:, :, -i2:, -j2:]

    final_feature = win3_feature
    final_feature = win3_feature + temp[:, :, :i3, :j3]
    final_feature = win3_feature + temp[:, :, :i3, -j3:]
    final_feature = win3_feature + temp[:, :, -i3:, :j3]
    final_feature = win3_feature + temp[:, :, -i3:, -j3:]

    final_feature = final_feature.reshape(num, -1)

    return final_feature
