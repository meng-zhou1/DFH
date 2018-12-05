
import cPickle as pickle

from hash.save_data import save_data
from libs.gcForest.lib.gcforest.gcforest import GCForest
from libs.gcForest.lib.gcforest.utils.config_utils import load_json


def cascade_forest(train_data=None, fg_label=None, updata_bit=None, ds=None):

    X_train = train_data['feat_data']
    y_train = train_data['label_data']

    fg_data = dict()
    fg_data['feat_data'] = train_data['feat_data']
    fg_data['label_data'] = fg_label

    print '\ncascade classification phase'

    ca_model = "./libs/gcForest/settings/{}/{}-ca.json".format(
        ds['name'], ds['name'])
    config = load_json(ca_model)
    gc = GCForest(config)

    fg_result_path = "./models/{}/fg/fg_tra_result.pkl".format(ds['name'])
    with open(fg_result_path, "rb") as f:
        fg_result = pickle.load(f)
    __ = gc.fit_transform(fg_result, y_train)
    print '\ncascade classification finished'

    # dump the cascade classification model
    ca_dump_path = "./models/{}/fg-ca/ca_{}.pkl".format(ds['name'], updata_bit)
    save_data(ca_dump_path, gc)
    print '\ncascade classification model saved'
    y_pred = gc.predict(fg_result)

    # # cascade only
    # ca_model = "./libs/gcForest/settings/{}/{}-ca_only.json".format(ds['name'], ds['name'])
    # config =load_json(ca_model)
    # gc = GCForest(config)
    # gc.fit_transform(X_train, y_train)
    # # dump the cascade classification model
    # ca_dump_path = "./models/{}/ca/ca_{}.pkl".format(ds['name'], updata_bit)
    # save_data(ca_dump_path, gc)
    # print '\ncascade classification model saved'
    # y_pred = gc.predict(X_train)

    return y_pred
